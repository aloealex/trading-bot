# monitoring/alerts.py
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class AlertSystem:
    """
    Sistema di allerta per notifiche via Telegram, Email, etc.
    """
    
    def __init__(self,
                 telegram_enabled: bool = False,
                 telegram_bot_token: Optional[str] = None,
                 telegram_chat_id: Optional[str] = None,
                 email_enabled: bool = False,
                 email_config: Optional[Dict] = None):
        
        self.telegram_enabled = telegram_enabled
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        
        self.email_enabled = email_enabled
        self.email_config = email_config or {}
        
        self.alerts_history = []
    
    def send_telegram(self, message: str) -> bool:
        """Invia un messaggio Telegram."""
        if not self.telegram_enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Errore invio Telegram: {e}")
            return False
    
    def send_email(self,
                   subject: str,
                   body: str,
                   recipient: Optional[str] = None) -> bool:
        """Invia un'email."""
        if not self.email_enabled:
            return False
        
        try:
            config = self.email_config
            recipient = recipient or config.get('recipient')
            
            if not recipient:
                return False
            
            msg = MIMEMultipart()
            msg['From'] = config.get('username')
            msg['To'] = recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config.get('smtp_server', 'smtp.gmail.com'), 
                                 config.get('smtp_port', 587))
            server.starttls()
            server.login(config.get('username'), config.get('password'))
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Errore invio email: {e}")
            return False
    
    def send_alert(self,
                   alert_type: str,
                   message: str,
                   level: str = 'info',
                   **kwargs) -> bool:
        """
        Invia un'alert.
        
        Args:
            alert_type: 'trade', 'signal', 'error', 'info'
            message: Messaggio dell'alert
            level: 'info', 'warning', 'error'
        """
        # Registra l'alert
        alert_entry = {
            'type': alert_type,
            'message': message,
            'level': level,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.alerts_history.append(alert_entry)
        
        # Formatta il messaggio
        formatted = f"🔔 <b>{alert_type.upper()}</b>\n{message}"
        
        if level == 'error':
            formatted = f"❌ {formatted}"
        elif level == 'warning':
            formatted = f"⚠️ {formatted}"
        else:
            formatted = f"ℹ️ {formatted}"
        
        # Invia via Telegram
        sent = False
        if self.telegram_enabled:
            sent = self.send_telegram(formatted)
        
        # Invia via Email solo per errori critici
        if level == 'error' and self.email_enabled:
            self.send_email(
                subject=f"[Trading Bot] {alert_type.upper()} Alert",
                body=message
            )
        
        return sent
    
    def send_trade_alert(self,
                         symbol: str,
                         side: str,
                         quantity: int,
                         price: float,
                         **kwargs):
        """Invia un alert per un trade."""
        message = (
            f"Trade: {symbol} {side.upper()} {quantity} shares @ ${price:.2f}\n"
            f"Value: ${quantity * price:,.2f}"
        )
        self.send_alert('trade', message, 'info', **kwargs)
    
    def send_signal_alert(self,
                          symbol: str,
                          signal: float,
                          confidence: float,
                          action: str,
                          **kwargs):
        """Invia un alert per un segnale."""
        emoji = '📈' if signal > 0 else '📉'
        message = (
            f"{emoji} Signal: {symbol} {action}\n"
            f"Gradimento: {signal:.3f}\n"
            f"Confidenza: {confidence:.1%}"
        )
        self.send_alert('signal', message, 'info', **kwargs)
    
    def send_error_alert(self, error: str, context: Optional[Dict] = None, **kwargs):
        """Invia un alert per un errore."""
        message = f"Error: {error}"
        if context:
            message += f"\nContext: {json.dumps(context, default=str)}"
        self.send_alert('error', message, 'error', **kwargs)
    
    def send_daily_summary(self, metrics: Dict[str, Any]):
        """Invia un riassunto giornaliero."""
        message = (
            f"📊 Daily Summary\n"
            f"Total Value: ${metrics.get('total_value', 0):,.2f}\n"
            f"Daily PnL: ${metrics.get('daily_pnl', 0):,.2f}\n"
            f"Positions: {metrics.get('positions_count', 0)}\n"
            f"Trades: {metrics.get('trades_count', 0)}\n"
            f"Win Rate: {metrics.get('win_rate', 0):.1%}"
        )
        self.send_alert('summary', message, 'info')