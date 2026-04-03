"""
FastAPI application for the Financial Market Environment.

Endpoints:
    POST /reset  — Reset the environment for a new episode
    POST /step   — Execute a trading action
    GET  /state  — Get current session state
    GET  /schema — Action / observation schemas
    WS   /ws     — WebSocket endpoint for persistent sessions (used by EnvClient)

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv-core is required. Install with: pip install 'openenv-core[core]'") from e

try:
    from ..models import MarketAction, MarketObservation
    from .market_environment import MarketEnvironment
except ImportError:
    from models import MarketAction, MarketObservation
    from server.market_environment import MarketEnvironment


app = create_app(
    MarketEnvironment,
    MarketAction,
    MarketObservation,
    env_name="financial_market_env",
    max_concurrent_envs=16,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
