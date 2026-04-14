import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run All-Rounder-Net over HTTPS")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--certfile", default="certs/localhost.pem")
    parser.add_argument("--keyfile", default="certs/localhost-key.pem")
    args = parser.parse_args()

    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        ssl_certfile=args.certfile,
        ssl_keyfile=args.keyfile,
    )


if __name__ == "__main__":
    main()
