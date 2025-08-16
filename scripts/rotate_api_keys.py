#!/usr/bin/env python3
"""
Script to handle API key rotation.
This script will:
1. Check for expiring API keys
2. Create new keys for clients with expiring keys
3. Notify clients about new keys
4. Revoke expired keys
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone
import logging
from typing import List, Dict
import aiosmtplib
from email.message import EmailMessage
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.core.security import api_key_manager
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_key_rotation.log')
    ]
)

class KeyRotationManager:
    def __init__(self):
        self.warning_days = 14  # Warn clients 14 days before expiration
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "api-support@example.com")
    
    async def get_expiring_keys(self) -> List[Dict]:
        """Get list of API keys that will expire soon."""
        warning_date = datetime.now(timezone.utc) + timedelta(days=self.warning_days)
        expiring_keys = []
        
        # In production, this would query a database
        # This is a simplified example
        async for key_data in self._get_all_keys():
            expires_at = datetime.fromisoformat(key_data["expires_at"])
            if expires_at <= warning_date:
                expiring_keys.append(key_data)
        
        return expiring_keys
    
    async def create_new_key(self, client_id: str) -> Dict:
        """Create a new API key for a client."""
        try:
            key_data = await api_key_manager.create_api_key(client_id)
            logger.info("Created new API key for client: %s", client_id)
            return key_data
        except Exception as e:
            logger.error("Failed to create new key for client %s: %s", client_id, str(e))
            raise
    
    async def notify_client(self, client_id: str, key_data: Dict) -> None:
        """Notify client about their new API key."""
        try:
            message = EmailMessage()
            message["From"] = self.from_email
            message["To"] = f"{client_id}@example.com"  # In production, get from client database
            message["Subject"] = "Your API Key is Expiring Soon"
            
            body = f"""
            Hello,
            
            Your Instagram Stats API key will expire on {key_data['expires_at']}.
            We have generated a new API key for you:
            
            New API Key: {key_data['api_key']}
            Expires At: {key_data['expires_at']}
            
            Please update your applications to use the new key before the expiration date.
            The old key will continue to work until its expiration date.
            
            Best regards,
            API Support Team
            """
            
            message.set_content(body)
            
            async with aiosmtplib.SMTP(
                hostname=self.smtp_host,
                port=self.smtp_port,
                use_tls=True
            ) as smtp:
                await smtp.login(self.smtp_user, self.smtp_password)
                await smtp.send_message(message)
            
            logger.info("Sent key rotation notification to client: %s", client_id)
        except Exception as e:
            logger.error("Failed to notify client %s: %s", client_id, str(e))
            raise
    
    async def revoke_expired_keys(self) -> None:
        """Revoke all expired API keys."""
        try:
            async for key_data in self._get_all_keys():
                expires_at = datetime.fromisoformat(key_data["expires_at"])
                if expires_at <= datetime.now(timezone.utc):
                    await api_key_manager.revoke_api_key(key_data["api_key"])
                    logger.info("Revoked expired key for client: %s", key_data['client_id'])
        except Exception as e:
            logger.error("Failed to revoke expired keys: %s", str(e))
            raise
    
    async def _get_all_keys(self):
        """Generator to get all API keys."""
        # In production, this would query a database
        # This is a simplified example that yields test data
        yield {
            "client_id": "test_client",
            "api_key": "test_key",
            "expires_at": (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        }

async def main():
    """Main key rotation process."""
    manager = KeyRotationManager()
    logger.info("Starting API key rotation process")
    
    try:
        # Get expiring keys
        expiring_keys = await manager.get_expiring_keys()
        logger.info("Found %d expiring keys", len(expiring_keys))
        
        # Process each expiring key
        for key_data in expiring_keys:
            client_id = key_data["client_id"]
            try:
                # Create new key
                new_key_data = await manager.create_new_key(client_id)
                
                # Notify client
                await manager.notify_client(client_id, new_key_data)
            except Exception as e:
                logger.error("Failed to process key rotation for client %s: %s", client_id, str(e))
                continue
        
        # Revoke expired keys
        await manager.revoke_expired_keys()
        
        logger.info("Completed API key rotation process")
    except Exception as e:
        logger.error("Key rotation process failed: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 