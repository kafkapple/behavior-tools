"""
Image scraper module for downloading images from web sources.
"""
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import hashlib
import warnings

# SSL 인증서 경고 무시
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class ImageScraper:
    """Base class for image scraping from various sources."""

    def __init__(self, config: dict, output_dir: str):
        """
        Initialize the image scraper.

        Args:
            config: Configuration dictionary with scraping parameters
            output_dir: Directory to save downloaded images
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_size = config.get('min_image_size', 224)
        self.max_size = config.get('max_image_size', 2048)
        self.timeout = config.get('timeout', 60)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.google.com/',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }

        self.downloaded_urls = set()
        self.failed_urls = []

    def download_image(self, url: str, save_path: Path, retry: int = 3) -> bool:
        """
        Download a single image from URL.

        Args:
            url: Image URL
            save_path: Path to save the image
            retry: Number of retry attempts

        Returns:
            True if successful, False otherwise
        """
        # URL 인코딩 - 공백 및 특수문자 처리
        from urllib.parse import quote
        try:
            # URL 검증 및 인코딩
            if ' ' in url or '\t' in url or '\n' in url:
                url = quote(url, safe=':/?#[]@!$&\'()*+,;=')
        except Exception as e:
            logger.debug(f"URL encoding failed: {e}")
            return False

        for attempt in range(retry):
            try:
                # SSL 검증 비활성화 옵션 추가 (인증서 만료 문제 해결)
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout,
                    stream=True,
                    verify=False,  # SSL 인증서 검증 비활성화
                    allow_redirects=True
                )
                response.raise_for_status()

                # Load and validate image
                img = Image.open(BytesIO(response.content))

                # Check image size
                width, height = img.size
                if width < self.min_size or height < self.min_size:
                    logger.debug(f"Image too small: {width}x{height}")
                    return False

                if width > self.max_size or height > self.max_size:
                    # Resize while maintaining aspect ratio
                    img.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)

                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save image
                img.save(save_path, 'JPEG', quality=95)
                self.downloaded_urls.add(url)
                return True

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    logger.debug(f"403 Forbidden (access denied): {url}")
                elif e.response.status_code == 404:
                    logger.debug(f"404 Not Found (broken link): {url}")
                else:
                    logger.debug(f"HTTP {e.response.status_code}: {url}")
                # 403, 404는 재시도 불필요
                if e.response.status_code in [403, 404]:
                    break
            except requests.exceptions.SSLError as e:
                logger.debug(f"SSL certificate error: {url}")
                # SSL 오류는 재시도 불필요
                break
            except requests.exceptions.Timeout:
                logger.debug(f"Timeout (attempt {attempt + 1}): {url}")
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed for {url}: {type(e).__name__}: {e}")
                time.sleep(1)

        self.failed_urls.append(url)
        return False

    def scrape(self, keywords: List[str], num_images_per_keyword: int) -> Dict[str, int]:
        """
        Scrape images for given keywords.

        Args:
            keywords: List of search keywords
            num_images_per_keyword: Number of images to download per keyword

        Returns:
            Dictionary with statistics
        """
        raise NotImplementedError("Subclasses must implement scrape method")


class BingImageScraper(ImageScraper):
    """Scraper using Bing Image Search (via bing-image-downloader)."""

    def scrape(self, keywords: List[str], num_images_per_keyword: int) -> Dict[str, int]:
        """Scrape images from Bing."""
        try:
            from bing_image_downloader import downloader
        except ImportError:
            logger.error("bing-image-downloader not installed. Install with: pip install bing-image-downloader")
            return {"total": 0, "failed": 0}

        total_downloaded = 0

        for keyword in tqdm(keywords, desc="Scraping keywords"):
            keyword_dir = self.output_dir / keyword.replace(" ", "_")

            try:
                downloader.download(
                    keyword,
                    limit=num_images_per_keyword,
                    output_dir=str(self.output_dir),
                    adult_filter_off=not self.config.get('adult_filter', True),
                    force_replace=False,
                    timeout=self.timeout,
                    verbose=False
                )

                # Count downloaded images
                if keyword_dir.exists():
                    count = len(list(keyword_dir.glob("*.jpg")) + list(keyword_dir.glob("*.png")))
                    total_downloaded += count
                    logger.info(f"Downloaded {count} images for '{keyword}'")

            except Exception as e:
                logger.error(f"Failed to scrape '{keyword}': {e}")

        return {
            "total": total_downloaded,
            "failed": len(self.failed_urls),
            "keywords": len(keywords)
        }


class ICrawlerScraper(ImageScraper):
    """Scraper using icrawler library (supports Google, Bing, Baidu)."""

    def __init__(self, config: dict, output_dir: str, engine: str = 'google'):
        """
        Initialize ICrawler scraper.

        Args:
            config: Configuration dictionary
            output_dir: Output directory for images
            engine: Search engine to use ('google', 'bing', 'baidu')
        """
        super().__init__(config, output_dir)
        self.engine = engine

    def scrape(self, keywords: List[str], num_images_per_keyword: int) -> Dict[str, int]:
        """Scrape images using icrawler."""
        try:
            from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
        except ImportError:
            logger.error("icrawler not installed. Install with: pip install icrawler")
            return {"total": 0, "failed": 0}

        total_downloaded = 0

        for keyword in tqdm(keywords, desc=f"Scraping with {self.engine}"):
            keyword_dir = self.output_dir / keyword.replace(" ", "_")
            keyword_dir.mkdir(parents=True, exist_ok=True)

            try:
                if self.engine.lower() == 'google':
                    crawler = GoogleImageCrawler(
                        storage={'root_dir': str(keyword_dir)},
                        log_level=logging.ERROR
                    )
                elif self.engine.lower() == 'bing':
                    crawler = BingImageCrawler(
                        storage={'root_dir': str(keyword_dir)},
                        log_level=logging.ERROR
                    )
                else:
                    logger.error(f"Unsupported engine: {self.engine}")
                    continue

                crawler.crawl(
                    keyword=keyword,
                    max_num=num_images_per_keyword,
                    min_size=(self.min_size, self.min_size),
                    max_size=(self.max_size, self.max_size)
                )

                # Count downloaded images
                count = len(list(keyword_dir.glob("*.jpg")) + list(keyword_dir.glob("*.png")))
                total_downloaded += count
                logger.info(f"Downloaded {count} images for '{keyword}'")

            except Exception as e:
                logger.error(f"Failed to scrape '{keyword}': {e}")

        return {
            "total": total_downloaded,
            "failed": len(self.failed_urls),
            "keywords": len(keywords)
        }


def create_scraper(config: dict, output_dir: str, engine: str = 'bing') -> ImageScraper:
    """
    Factory function to create appropriate scraper.

    Args:
        config: Configuration dictionary
        output_dir: Output directory
        engine: Scraper engine ('bing', 'google', 'icrawler')

    Returns:
        ImageScraper instance
    """
    if engine.lower() == 'bing':
        return BingImageScraper(config, output_dir)
    elif engine.lower() in ['google', 'icrawler']:
        return ICrawlerScraper(config, output_dir, engine='google')
    else:
        logger.warning(f"Unknown engine '{engine}', defaulting to Bing")
        return BingImageScraper(config, output_dir)
