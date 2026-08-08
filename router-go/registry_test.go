package main

// Registry tests — both backends, no bucket and no network.
//
// The S3 path is driven through the S3API interface by a fake that serves
// paginated keys from memory, so ListObjectsV2/GetObject behavior (pagination,
// non-.json keys, torn bodies, unreadable objects) is verified offline.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

func heartbeat(t *testing.T, workerID, addr string, lastSeen float64) string {
	t.Helper()
	doc, err := json.Marshal(map[string]any{
		"worker_id":       workerID,
		"addr":            addr,
		"model":           "shakespeare",
		"market":          "local",
		"requests_served": 7,
		"last_seen":       lastSeen,
	})
	if err != nil {
		t.Fatalf("marshal heartbeat: %v", err)
	}
	return string(doc)
}

func writeFile(t *testing.T, dir, name, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

func ids(docs []WorkerDoc) []string {
	out := make([]string, len(docs))
	for i, d := range docs {
		out[i] = d.WorkerID
	}
	return out
}

func TestDirRegistrySkipsTornAndForeignFiles(t *testing.T) {
	dir := t.TempDir()
	now := float64(time.Now().Unix())
	writeFile(t, dir, "w0.json", heartbeat(t, "w0", "127.0.0.1:8001", now))
	writeFile(t, dir, "w1.json", heartbeat(t, "w1", "127.0.0.1:8002", now))
	writeFile(t, dir, "w2.json", `{"worker_id": "w2", "addr": `) // torn write
	writeFile(t, dir, "notes.txt", "not a heartbeat at all")
	if err := os.Mkdir(filepath.Join(dir, "nested.json"), 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	docs, err := DirRegistry{Dir: dir}.ListWorkers(context.Background())
	if err != nil {
		t.Fatalf("ListWorkers: %v", err)
	}
	if got, want := strings.Join(ids(docs), ","), "w0,w1"; got != want {
		t.Errorf("docs = %s, want %s (torn + non-json skipped)", got, want)
	}
	if docs[0].Addr != "127.0.0.1:8001" || docs[0].RequestsServed != 7 {
		t.Errorf("doc not decoded: %+v", docs[0])
	}
}

func TestDirRegistryMissingDirIsEmpty(t *testing.T) {
	for _, dir := range []string{filepath.Join(t.TempDir(), "nope"), ""} {
		docs, err := DirRegistry{Dir: dir}.ListWorkers(context.Background())
		if err != nil {
			t.Fatalf("ListWorkers(%q): %v", dir, err)
		}
		if len(docs) != 0 {
			t.Errorf("ListWorkers(%q) = %v, want empty", dir, ids(docs))
		}
	}
}

func TestLiveWorkersFiltersByTTLAndAddr(t *testing.T) {
	now := time.Unix(1_000_000, 0)
	nowSec := float64(now.Unix())
	docs := []WorkerDoc{
		{WorkerID: "fresh", Addr: "a:1", LastSeen: nowSec - 1},
		{WorkerID: "at-ttl", Addr: "b:1", LastSeen: nowSec - 15}, // boundary: inclusive
		{WorkerID: "stale", Addr: "c:1", LastSeen: nowSec - 15.5},
		{WorkerID: "ancient", Addr: "d:1", LastSeen: 0},
		{WorkerID: "no-addr", Addr: "", LastSeen: nowSec},
		{WorkerID: "clock-skew-ahead", Addr: "e:1", LastSeen: nowSec + 2},
	}

	live := LiveWorkers(docs, DefaultTTLSeconds, now)
	if got, want := strings.Join(ids(live), ","), "fresh,at-ttl,clock-skew-ahead"; got != want {
		t.Errorf("live = %s, want %s", got, want)
	}
}

func TestLiveWorkersEmptyIsNotNil(t *testing.T) {
	live := LiveWorkers(nil, DefaultTTLSeconds, time.Now())
	if live == nil {
		t.Fatal("LiveWorkers(nil) = nil; /fleet/status would emit null instead of []")
	}
}

// /fleet/status must report exactly what the worker wrote, extra fields and
// all — the Python router hands back the raw dict.
func TestWorkerDocPreservesUnknownFields(t *testing.T) {
	raw := `{"worker_id":"w0","addr":"h:1","last_seen":12.5,"gpu":"A10G","zone":"us-east-1a"}`
	var doc WorkerDoc
	if err := json.Unmarshal([]byte(raw), &doc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if doc.WorkerID != "w0" || doc.Addr != "h:1" || doc.LastSeen != 12.5 {
		t.Fatalf("typed fields wrong: %+v", doc)
	}
	out, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(out) != raw {
		t.Errorf("round trip = %s, want %s", out, raw)
	}

	// A doc built in code (no raw bytes) still marshals sensibly.
	out, err = json.Marshal(WorkerDoc{WorkerID: "w1", Addr: "h:2", LastSeen: 1})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if want := `{"worker_id":"w1","addr":"h:2","last_seen":1}`; string(out) != want {
		t.Errorf("marshal = %s, want %s", out, want)
	}
}

// fakeS3 serves paginated keys and bodies from memory. It implements S3API,
// which is exactly the surface S3Registry uses.
type fakeS3 struct {
	pages   [][]string        // keys, one slice per page
	bodies  map[string]string // key -> object body
	getErrs map[string]error  // key -> failure to inject
	listErr error
	gets    []string
}

func (f *fakeS3) ListObjectsV2(
	_ context.Context, in *s3.ListObjectsV2Input, _ ...func(*s3.Options),
) (*s3.ListObjectsV2Output, error) {
	if f.listErr != nil {
		return nil, f.listErr
	}
	page := 0
	if in.ContinuationToken != nil {
		page, _ = strconv.Atoi(*in.ContinuationToken)
	}
	out := &s3.ListObjectsV2Output{}
	for _, key := range f.pages[page] {
		if strings.HasPrefix(key, aws.ToString(in.Prefix)) {
			out.Contents = append(out.Contents, types.Object{Key: aws.String(key)})
		}
	}
	if page+1 < len(f.pages) {
		out.IsTruncated = aws.Bool(true)
		out.NextContinuationToken = aws.String(strconv.Itoa(page + 1))
	}
	return out, nil
}

func (f *fakeS3) GetObject(
	_ context.Context, in *s3.GetObjectInput, _ ...func(*s3.Options),
) (*s3.GetObjectOutput, error) {
	key := aws.ToString(in.Key)
	f.gets = append(f.gets, key)
	if err, ok := f.getErrs[key]; ok {
		return nil, err
	}
	body, ok := f.bodies[key]
	if !ok {
		return nil, fmt.Errorf("NoSuchKey: %s", key)
	}
	return &s3.GetObjectOutput{Body: io.NopCloser(strings.NewReader(body))}, nil
}

func TestS3RegistryListsAcrossPages(t *testing.T) {
	now := float64(time.Now().Unix())
	fake := &fakeS3{
		pages: [][]string{
			{"fleet/f1/workers/w0.json", "fleet/f1/workers/w1.json", "fleet/f1/workers/_SUCCESS"},
			{"fleet/f1/workers/w2.json", "fleet/f1/workers/torn.json", "fleet/f1/workers/gone.json"},
		},
		bodies: map[string]string{
			"fleet/f1/workers/w0.json":   heartbeat(t, "w0", "10.0.0.1:8001", now),
			"fleet/f1/workers/w1.json":   heartbeat(t, "w1", "10.0.0.2:8001", now),
			"fleet/f1/workers/w2.json":   heartbeat(t, "w2", "10.0.0.3:8001", now),
			"fleet/f1/workers/torn.json": `{"worker_id": "torn"`, // half-written
			"fleet/f1/workers/_SUCCESS":  "",
		},
		getErrs: map[string]error{
			"fleet/f1/workers/gone.json": errors.New("NoSuchKey"),
		},
	}

	reg := NewS3Registry(fake, "bucket", "fleet/f1/workers")
	docs, err := reg.ListWorkers(context.Background())
	if err != nil {
		t.Fatalf("ListWorkers: %v", err)
	}
	if got, want := strings.Join(ids(docs), ","), "w0,w1,w2"; got != want {
		t.Errorf("docs = %s, want %s", got, want)
	}
	for _, key := range fake.gets {
		if !strings.HasSuffix(key, ".json") {
			t.Errorf("fetched %q — non-.json keys must be skipped before GetObject", key)
		}
	}
}

func TestS3RegistryPropagatesListErrors(t *testing.T) {
	fake := &fakeS3{pages: [][]string{{}}, listErr: errors.New("AccessDenied")}
	if _, err := NewS3Registry(fake, "bucket", "p/").ListWorkers(context.Background()); err == nil {
		t.Fatal("ListWorkers succeeded, want the AccessDenied surfaced so the poll logs it")
	}
}

// A prefix must be a directory prefix, so "…/workers" cannot also match
// "…/workers-old/…".
func TestS3PrefixNormalization(t *testing.T) {
	fake := &fakeS3{
		pages: [][]string{{
			"fleet/f1/workers/w0.json",
			"fleet/f1/workers-old/w9.json",
		}},
		bodies: map[string]string{
			"fleet/f1/workers/w0.json":     heartbeat(t, "w0", "10.0.0.1:8001", float64(time.Now().Unix())),
			"fleet/f1/workers-old/w9.json": heartbeat(t, "w9", "10.0.0.9:8001", float64(time.Now().Unix())),
		},
	}

	docs, err := NewS3Registry(fake, "bucket", "fleet/f1/workers").ListWorkers(context.Background())
	if err != nil {
		t.Fatalf("ListWorkers: %v", err)
	}
	if got, want := strings.Join(ids(docs), ","), "w0"; got != want {
		t.Errorf("docs = %s, want %s", got, want)
	}
}

func TestParseS3URI(t *testing.T) {
	cases := []struct {
		uri            string
		bucket, prefix string
		ok             bool
	}{
		{"s3://bucket/fleet/f1/workers", "bucket", "fleet/f1/workers", true},
		{"s3://bucket/fleet/f1/workers/", "bucket", "fleet/f1/workers/", true},
		{"s3://bucket", "bucket", "", true},
		{"/tmp/.fleet/local/workers", "", "", false},
		{"", "", "", false},
	}
	for _, tc := range cases {
		bucket, prefix, ok := parseS3URI(tc.uri)
		if ok != tc.ok || bucket != tc.bucket || prefix != tc.prefix {
			t.Errorf("parseS3URI(%q) = (%q, %q, %v), want (%q, %q, %v)",
				tc.uri, bucket, prefix, ok, tc.bucket, tc.prefix, tc.ok)
		}
	}
}

func TestNewRegistryPicksDirBackend(t *testing.T) {
	reg, err := NewRegistry(context.Background(), t.TempDir())
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}
	if _, ok := reg.(DirRegistry); !ok {
		t.Errorf("NewRegistry(dir) = %T, want DirRegistry", reg)
	}
}
