# Fleet router — Go, multi-stage, distroless.
#
# The router is pure I/O (reverse proxy + retry policy + registry watch), so it
# needs no interpreter, no libc extras and no shell. A distroless static base
# gives a ~15MB image with essentially no attack surface: nothing to exec into
# if the process is ever compromised.
#
# ⚠️ Cross-arch: build cloud images with --platform linux/amd64 (see build.sh).
# TARGETOS/TARGETARCH are set by buildx and drive the cross-compile below.

FROM golang:1.22-alpine AS build
WORKDIR /src

# Module files first: dependencies re-download only when they actually change,
# not on every source edit.
COPY router-go/go.mod router-go/go.sum ./
RUN go mod download

COPY router-go/ ./

# CGO_ENABLED=0 is what makes the static base viable -- with cgo the binary
# would need a libc the distroless static image does not ship.
# -trimpath and -ldflags="-s -w" drop local paths and debug symbols: smaller
# image, and no build-machine paths leaking into a shipped artifact.
ARG TARGETOS=linux
ARG TARGETARCH
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags="-s -w" -o /out/router .

# --- runtime -------------------------------------------------------------
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/router /router

# nonroot (uid 65532) is baked into the base image; the k8s manifests assert
# runAsNonRoot, so this must not be overridden.
USER nonroot:nonroot
EXPOSE 8000

# No HEALTHCHECK: Kubernetes probes /healthz. Distroless has no shell anyway,
# so a docker-level healthcheck could not run.
ENTRYPOINT ["/router"]
