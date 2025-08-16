#!/usr/bin/env python3
"""
Backup script for Instagram Stats API.
This script will:
1. Create a backup of Redis data
2. Compress and archive logs
3. Create a backup of API keys and configurations
4. Upload backups to remote storage (if configured)
"""

import os
import sys
import shutil
import logging
import asyncio
import aiohttp
from datetime import datetime, timezone
from pathlib import Path
import tarfile
import redis
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backup.log')
    ]
)

class BackupManager:
    def __init__(self):
        self.backup_dir = Path("/opt/instagram_stats_api/backups")
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True
        )
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.remote_storage_url = os.getenv("REMOTE_STORAGE_URL")
        self.remote_storage_key = os.getenv("REMOTE_STORAGE_KEY")
    
    def backup_redis(self) -> Optional[Path]:
        """Create Redis backup."""
        try:
            backup_path = self.backup_dir / f"redis_{self.timestamp}.rdb"
            self.redis_client.save()
            shutil.copy("/var/lib/redis/dump.rdb", backup_path)
            logger.info(f"Redis backup created at {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Redis backup failed: {str(e)}")
            return None
    
    def backup_logs(self) -> Optional[Path]:
        """Archive log files."""
        try:
            log_dir = Path("/var/log/instagram_api")
            if not log_dir.exists():
                logger.warning("Log directory not found")
                return None
            
            archive_path = self.backup_dir / f"logs_{self.timestamp}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(log_dir, arcname=log_dir.name)
            
            logger.info(f"Logs archived at {archive_path}")
            return archive_path
        except Exception as e:
            logger.error(f"Log archival failed: {str(e)}")
            return None
    
    def backup_configs(self) -> Optional[Path]:
        """Backup configuration files."""
        try:
            config_dir = Path("/opt/instagram_stats_api/config")
            if not config_dir.exists():
                logger.warning("Config directory not found")
                return None
            
            archive_path = self.backup_dir / f"configs_{self.timestamp}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(config_dir, arcname=config_dir.name)
            
            logger.info(f"Configs backed up at {archive_path}")
            return archive_path
        except Exception as e:
            logger.error(f"Config backup failed: {str(e)}")
            return None
    
    async def upload_to_remote(self, file_path: Path) -> bool:
        """Upload backup to remote storage."""
        if not self.remote_storage_url or not self.remote_storage_key:
            logger.warning("Remote storage not configured")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.remote_storage_key}"}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                with open(file_path, "rb") as f:
                    data = aiohttp.FormData()
                    data.add_field("file",
                                 f,
                                 filename=file_path.name,
                                 content_type="application/octet-stream")
                    
                    async with session.post(
                        f"{self.remote_storage_url}/upload",
                        data=data,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Uploaded {file_path.name} to remote storage")
                            return True
                        else:
                            logger.error(f"Upload failed: {await response.text()}")
                            return False
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            return False
    
    def cleanup_old_backups(self, days: int = 7) -> None:
        """Remove backups older than specified days."""
        try:
            for backup_file in self.backup_dir.glob("*"):
                if backup_file.stat().st_mtime < (datetime.now(timezone.utc).timestamp() - days * 86400):
                    backup_file.unlink()
                    logger.info(f"Removed old backup: {backup_file}")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

async def main():
    """Main backup process."""
    backup_manager = BackupManager()
    logger.info("Starting backup process")
    
    try:
        # Create backup directory if it doesn't exist
        backup_manager.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Perform backups
        backup_files = []
        if redis_backup := backup_manager.backup_redis():
            backup_files.append(redis_backup)
        
        if logs_backup := backup_manager.backup_logs():
            backup_files.append(logs_backup)
        
        if config_backup := backup_manager.backup_configs():
            backup_files.append(config_backup)
        
        # Upload to remote storage if configured
        for backup_file in backup_files:
            await backup_manager.upload_to_remote(backup_file)
        
        # Cleanup old backups
        backup_manager.cleanup_old_backups()
        
        logger.info("Backup process completed successfully")
    except Exception as e:
        logger.error(f"Backup process failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 