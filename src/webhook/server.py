"""TradingView webhook server."""

import asyncio
import hmac
import hashlib
from typing import Optional, Callable, Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

from utils.logger import get_logger

logger = get_logger("webhook")


class WebhookAlert(BaseModel):
    """TradingView webhook alert payload."""
    action: str  # "buy" or "sell"
    symbol: str
    price: float
    time: Optional[str] = None
    message: Optional[str] = None
    # VMC-specific fields
    wt1: Optional[float] = None
    wt2: Optional[float] = None
    vwap: Optional[float] = None
    mfi: Optional[float] = None


class WebhookServer:
    """
    TradingView webhook server.

    Receives alerts from TradingView and forwards them to the trading bot.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        secret: str = "",
        path: str = "/webhook"
    ):
        """
        Initialize webhook server.

        Args:
            host: Server host
            port: Server port
            secret: Secret for webhook validation
            path: Webhook endpoint path
        """
        self.host = host
        self.port = port
        self.secret = secret
        self.path = path

        self.app = FastAPI(title="VMC Trading Bot Webhook")
        self._callback: Optional[Callable] = None
        self._setup_routes()

    def set_callback(self, callback: Callable[[WebhookAlert], Any]) -> None:
        """
        Set callback function for received alerts.

        Args:
            callback: Async function to call with alert data
        """
        self._callback = callback

    def _setup_routes(self) -> None:
        """Set up FastAPI routes."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "ok", "time": datetime.utcnow().isoformat()}

        @self.app.post(self.path)
        async def webhook_handler(
            request: Request,
            background_tasks: BackgroundTasks
        ):
            """Handle incoming webhook alerts."""
            try:
                # Get raw body for signature verification
                body = await request.body()

                # Verify signature if secret is set
                if self.secret:
                    signature = request.headers.get("X-Webhook-Signature", "")
                    if not self._verify_signature(body, signature):
                        logger.warning("Invalid webhook signature")
                        raise HTTPException(status_code=401, detail="Invalid signature")

                # Parse JSON body
                try:
                    data = await request.json()
                except Exception:
                    # Try to parse as plain text (TradingView sometimes sends plain text)
                    text = body.decode('utf-8')
                    data = self._parse_text_alert(text)

                logger.info(f"Received webhook: {data}")

                # Validate and create alert
                alert = self._create_alert(data)

                # Process alert in background
                if self._callback:
                    background_tasks.add_task(self._process_alert, alert)

                return {"status": "ok", "received": alert.dict()}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                raise HTTPException(status_code=400, detail=str(e))

    def _verify_signature(self, body: bytes, signature: str) -> bool:
        """
        Verify webhook signature.

        Args:
            body: Request body
            signature: Provided signature

        Returns:
            True if signature is valid
        """
        if not self.secret:
            return True

        expected = hmac.new(
            self.secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected)

    def _parse_text_alert(self, text: str) -> Dict[str, Any]:
        """
        Parse plain text alert from TradingView.

        Expected format:
        action=buy
        symbol=BTC
        price=50000
        ...

        Args:
            text: Plain text alert

        Returns:
            Parsed data dictionary
        """
        data = {}
        for line in text.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lower()
                value = value.strip()

                # Try to convert numeric values
                try:
                    if '.' in value:
                        data[key] = float(value)
                    else:
                        data[key] = int(value)
                except ValueError:
                    data[key] = value

        return data

    def _create_alert(self, data: Dict[str, Any]) -> WebhookAlert:
        """
        Create WebhookAlert from data.

        Args:
            data: Alert data dictionary

        Returns:
            WebhookAlert object
        """
        # Map common field variations
        action = data.get('action') or data.get('side') or data.get('direction', 'unknown')
        symbol = data.get('symbol') or data.get('ticker', 'UNKNOWN')
        price = float(data.get('price') or data.get('close', 0))

        return WebhookAlert(
            action=action.lower(),
            symbol=symbol.upper(),
            price=price,
            time=data.get('time'),
            message=data.get('message'),
            wt1=data.get('wt1'),
            wt2=data.get('wt2'),
            vwap=data.get('vwap'),
            mfi=data.get('mfi')
        )

    async def _process_alert(self, alert: WebhookAlert) -> None:
        """
        Process received alert.

        Args:
            alert: Webhook alert to process
        """
        if self._callback:
            try:
                result = self._callback(alert)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error processing alert: {e}")

    def run(self) -> None:
        """Run the webhook server (blocking)."""
        logger.info(f"Starting webhook server on {self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )

    async def start_async(self) -> None:
        """Start the webhook server asynchronously."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
