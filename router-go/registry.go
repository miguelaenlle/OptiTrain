package main

// Worker registry — the read side of src/inference/registry.py.
//
// Workers overwrite <workers_uri>/<worker_id>.json every few seconds; the
// router lists the prefix and treats a worker as live while its last_seen is
// inside the TTL. Two backends behind one interface:
//
//	Dir — a local directory of JSON docs (what `fleet up --local` and k3d use)
//	S3  — ListObjectsV2 + GetObject, taken as an INTERFACE so the unit tests
//	      run against a fake and never touch a real bucket.
//
// Torn or non-JSON documents are skipped rather than failing the sweep: the
// next poll heals it, and a half-written heartbeat must never blank the fleet.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// DefaultTTLSeconds mirrors registry.DEFAULT_TTL_SECONDS (~3x the heartbeat
// interval, so normal NTP skew never evicts a healthy worker).
const DefaultTTLSeconds = 15.0

const s3Scheme = "s3://"

// WorkerDoc is one heartbeat document. Unknown fields are preserved verbatim
// (see MarshalJSON) so /fleet/status reports exactly what the worker wrote,
// the same way the Python router hands back the raw dict.
type WorkerDoc struct {
	WorkerID       string  `json:"worker_id"`
	Addr           string  `json:"addr"` // "host:port" the router dials
	Model          string  `json:"model,omitempty"`
	Market         string  `json:"market,omitempty"`
	RequestsServed int     `json:"requests_served,omitempty"`
	LastSeen       float64 `json:"last_seen"`

	raw json.RawMessage // the bytes as written, when this came off the wire
}

// UnmarshalJSON decodes the typed fields and keeps the original bytes.
func (d *WorkerDoc) UnmarshalJSON(b []byte) error {
	type plain WorkerDoc
	var p plain
	if err := json.Unmarshal(b, &p); err != nil {
		return err
	}
	*d = WorkerDoc(p)
	d.raw = append(json.RawMessage(nil), b...)
	return nil
}

// MarshalJSON re-emits the original document when there is one, so extra
// fields a worker added survive the round trip through /fleet/status.
func (d WorkerDoc) MarshalJSON() ([]byte, error) {
	if len(d.raw) > 0 {
		return d.raw, nil
	}
	type plain WorkerDoc
	return json.Marshal(plain(d))
}

// Registry is every way the router can discover workers. In-cluster this gains
// a third implementation (a K8s Endpoints watch) with no change above it.
type Registry interface {
	// ListWorkers returns all heartbeat documents under the prefix, live and
	// stale alike. Liveness filtering is LiveWorkers' job.
	ListWorkers(ctx context.Context) ([]WorkerDoc, error)
}

// LiveWorkers keeps workers whose heartbeat is fresher than the TTL and that
// advertise an address. Same two rules as registry.live_workers.
func LiveWorkers(docs []WorkerDoc, ttlSeconds float64, now time.Time) []WorkerDoc {
	nowSec := float64(now.UnixNano()) / 1e9
	live := make([]WorkerDoc, 0, len(docs))
	for _, d := range docs {
		if nowSec-d.LastSeen <= ttlSeconds && d.Addr != "" {
			live = append(live, d)
		}
	}
	return live
}

// DirRegistry reads heartbeat docs from a local directory.
type DirRegistry struct {
	Dir string
}

// ListWorkers implements Registry.
func (r DirRegistry) ListWorkers(_ context.Context) ([]WorkerDoc, error) {
	entries, err := os.ReadDir(r.Dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // Python: `if not os.path.isdir(...): return []`
		}
		return nil, err
	}
	docs := make([]WorkerDoc, 0, len(entries)) // os.ReadDir is name-sorted
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		body, err := os.ReadFile(filepath.Join(r.Dir, e.Name()))
		if err != nil {
			continue
		}
		var doc WorkerDoc
		if json.Unmarshal(body, &doc) != nil {
			continue // torn write / non-doc file: skip, next poll heals
		}
		docs = append(docs, doc)
	}
	return docs, nil
}

// S3API is the slice of the S3 client the registry uses. *s3.Client satisfies
// it, and so does the fake in registry_test.go — no bucket is ever needed to
// test this code.
type S3API interface {
	ListObjectsV2(ctx context.Context, params *s3.ListObjectsV2Input, optFns ...func(*s3.Options)) (*s3.ListObjectsV2Output, error)
	GetObject(ctx context.Context, params *s3.GetObjectInput, optFns ...func(*s3.Options)) (*s3.GetObjectOutput, error)
}

// S3Registry reads heartbeat docs from an S3 prefix.
type S3Registry struct {
	API    S3API
	Bucket string
	Prefix string // already normalized to end with "/" (or empty for the root)
}

// NewS3Registry builds a registry over any S3API — the seam the tests use.
func NewS3Registry(api S3API, bucket, prefix string) *S3Registry {
	return &S3Registry{API: api, Bucket: bucket, Prefix: normalizeS3Prefix(prefix)}
}

// ListWorkers implements Registry.
func (r *S3Registry) ListWorkers(ctx context.Context) ([]WorkerDoc, error) {
	pager := s3.NewListObjectsV2Paginator(r.API, &s3.ListObjectsV2Input{
		Bucket: aws.String(r.Bucket),
		Prefix: aws.String(r.Prefix),
	})
	var docs []WorkerDoc
	for pager.HasMorePages() {
		page, err := pager.NextPage(ctx)
		if err != nil {
			return nil, fmt.Errorf("list %s%s/%s: %w", s3Scheme, r.Bucket, r.Prefix, err)
		}
		for _, obj := range page.Contents {
			key := aws.ToString(obj.Key)
			if !strings.HasSuffix(key, ".json") {
				continue
			}
			doc, err := r.getDoc(ctx, key)
			if err != nil {
				continue // torn write / unreadable object: skip
			}
			docs = append(docs, doc)
		}
	}
	return docs, nil
}

func (r *S3Registry) getDoc(ctx context.Context, key string) (WorkerDoc, error) {
	var doc WorkerDoc
	out, err := r.API.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(r.Bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return doc, err
	}
	defer out.Body.Close()
	body, err := io.ReadAll(out.Body)
	if err != nil {
		return doc, err
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return doc, err
	}
	return doc, nil
}

// NewRegistry picks a backend from the URI: "s3://bucket/prefix" or a local
// directory. An empty URI yields a directory registry that finds nothing,
// which is exactly what the Python router does (and it warns at startup).
func NewRegistry(ctx context.Context, uri string) (Registry, error) {
	bucket, prefix, ok := parseS3URI(uri)
	if !ok {
		return DirRegistry{Dir: uri}, nil
	}
	cfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("load aws config: %w", err)
	}
	return NewS3Registry(s3.NewFromConfig(cfg), bucket, prefix), nil
}

func parseS3URI(uri string) (bucket, prefix string, ok bool) {
	if !strings.HasPrefix(uri, s3Scheme) {
		return "", "", false
	}
	bucket, prefix, _ = strings.Cut(strings.TrimPrefix(uri, s3Scheme), "/")
	return bucket, prefix, bucket != ""
}

// normalizeS3Prefix makes the prefix a directory prefix, so "fleet/f1/workers"
// cannot also match "fleet/f1/workers-old/...".
func normalizeS3Prefix(prefix string) string {
	prefix = strings.TrimRight(prefix, "/")
	if prefix == "" {
		return ""
	}
	return prefix + "/"
}
