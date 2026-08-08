# Stub worker — answers /v1/completions instantly, runs no model.
#
# Used for the distributed-systems experiments (cross-node routing, node kill,
# the black-hole failure mode). Those test the ROUTER and the CLUSTER, not the
# model, and a stub isolates that: upstream latency ~0, so anything measured is
# infrastructure. It also cross-compiles to amd64 in seconds, where the real
# torch worker would need ~1.2GB built under QEMU emulation on an arm64 Mac.
#
# The real worker image (worker.Dockerfile) is what Stage 2 runs on GPU nodes.

FROM golang:1.22-alpine AS build
WORKDIR /src
COPY bench/stubworker/go.mod ./
RUN go mod download
COPY bench/stubworker/ ./
ARG TARGETOS=linux
ARG TARGETARCH
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags="-s -w" -o /out/stubworker .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/stubworker /stubworker
USER nonroot:nonroot
EXPOSE 8001
ENTRYPOINT ["/stubworker"]
