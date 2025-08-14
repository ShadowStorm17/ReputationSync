"""
SEO plugin.
Integrates SEO analysis with reputation monitoring.
"""

from typing import Dict, List, Any, Optional
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from app.core.plugins.base import SEOPlugin, PluginType, PluginMetadata
from app.core.error_handling import ReputationError, ErrorSeverity, ErrorCategory

class ReputationSEO(SEOPlugin):
    """Plugin for SEO analysis and reputation monitoring integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SEO plugin."""
        super().__init__(config)
        self._session: Optional[aiohttp.ClientSession] = None
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="reputation_seo",
            version="1.0.0",
            description="SEO plugin for reputation monitoring integration",
            author="Reputation Sync Team",
            type=PluginType.SEO,
            config_schema={
                "type": "object",
                "properties": {
                    "search_engines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search engines to monitor"
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords to track"
                    },
                    "competitors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Competitor domains to monitor"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "SEO metrics to track"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "HTTP request timeout in seconds",
                        "default": 30
                    }
                }
            }
        )
    
    async def initialize(self) -> bool:
        """Initialize plugin."""
        try:
            if self._session is None:
                timeout = aiohttp.ClientTimeout(total=self.config.get("timeout", 30))
                self._session = aiohttp.ClientSession(timeout=timeout)
            return True
        except Exception as e:
            raise ReputationError(
                message=f"Error initializing SEO plugin: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def shutdown(self) -> bool:
        """Shutdown plugin."""
        try:
            if self._session is not None:
                await self._session.close()
                self._session = None
            return True
        except Exception as e:
            raise ReputationError(
                message=f"Error shutting down SEO plugin: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def analyze_url(
        self,
        url: str
    ) -> Dict[str, Any]:
        """Analyze SEO metrics for a URL."""
        try:
            if self._session is None:
                raise ReputationError(
                    message="Plugin not initialized",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ReputationError(
                    message="Invalid URL",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Fetch page content
            async with self._session.get(url) as response:
                if response.status != 200:
                    raise ReputationError(
                        message=f"Error fetching URL: {response.status}",
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.BUSINESS
                    )
                
                content = await response.text()
            
            # Parse content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract SEO metrics
            metrics = {
                "title": self._extract_title(soup),
                "meta_description": self._extract_meta_description(soup),
                "headings": self._extract_headings(soup),
                "images": self._extract_images(soup),
                "links": self._extract_links(soup),
                "content_length": len(soup.get_text()),
                "load_time": response.elapsed.total_seconds()
            }
            
            return metrics
            
        except Exception as e:
            raise ReputationError(
                message=f"Error analyzing URL: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def track_keyword_rankings(
        self,
        keyword: str,
        domain: str
    ) -> Dict[str, Any]:
        """Track keyword rankings across search engines."""
        try:
            rankings = {}
            
            for engine in self.config.get("search_engines", []):
                # Simulate search engine API call
                ranking = await self._get_keyword_ranking(engine, keyword, domain)
                rankings[engine] = ranking
            
            return {
                "keyword": keyword,
                "domain": domain,
                "rankings": rankings
            }
            
        except Exception as e:
            raise ReputationError(
                message=f"Error tracking keyword rankings: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def analyze_competitor(
        self,
        competitor_domain: str
    ) -> Dict[str, Any]:
        """Analyze competitor's SEO performance."""
        try:
            # Validate domain
            parsed_domain = urlparse(competitor_domain)
            if not parsed_domain.netloc:
                raise ReputationError(
                    message="Invalid domain",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Analyze competitor's homepage
            homepage_url = f"https://{parsed_domain.netloc}"
            homepage_metrics = await self.analyze_url(homepage_url)
            
            # Track keyword rankings
            keyword_rankings = {}
            for keyword in self.config.get("keywords", []):
                ranking = await self.track_keyword_rankings(keyword, competitor_domain)
                keyword_rankings[keyword] = ranking
            
            return {
                "domain": competitor_domain,
                "homepage_metrics": homepage_metrics,
                "keyword_rankings": keyword_rankings
            }
            
        except Exception as e:
            raise ReputationError(
                message=f"Error analyzing competitor: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def get_reputation_impact(
        self,
        url: str
    ) -> Dict[str, Any]:
        """Get SEO impact on reputation."""
        try:
            # Analyze URL
            seo_metrics = await self.analyze_url(url)
            
            # Calculate reputation impact
            impact = {
                "visibility_score": self._calculate_visibility_score(seo_metrics),
                "content_quality_score": self._calculate_content_quality_score(seo_metrics),
                "user_experience_score": self._calculate_user_experience_score(seo_metrics),
                "overall_score": 0.0
            }
            
            # Calculate overall score
            impact["overall_score"] = (
                impact["visibility_score"] * 0.4 +
                impact["content_quality_score"] * 0.3 +
                impact["user_experience_score"] * 0.3
            )
            
            return {
                "url": url,
                "seo_metrics": seo_metrics,
                "impact": impact
            }
            
        except Exception as e:
            raise ReputationError(
                message=f"Error getting reputation impact: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    def _extract_title(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract and analyze page title."""
        title = soup.find('title')
        if not title:
            return {"text": "", "length": 0, "has_keywords": False}
        
        title_text = title.get_text().strip()
        return {
            "text": title_text,
            "length": len(title_text),
            "has_keywords": any(
                keyword.lower() in title_text.lower()
                for keyword in self.config.get("keywords", [])
            )
        }
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract and analyze meta description."""
        meta = soup.find('meta', attrs={'name': 'description'})
        if not meta:
            return {"text": "", "length": 0, "has_keywords": False}
        
        description = meta.get('content', '').strip()
        return {
            "text": description,
            "length": len(description),
            "has_keywords": any(
                keyword.lower() in description.lower()
                for keyword in self.config.get("keywords", [])
            )
        }
    
    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract and analyze headings."""
        headings = {
            "h1": [],
            "h2": [],
            "h3": []
        }
        
        for level in headings:
            for heading in soup.find_all(level):
                text = heading.get_text().strip()
                if text:
                    headings[level].append(text)
        
        return headings
    
    def _extract_images(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract and analyze images."""
        images = []
        
        for img in soup.find_all('img'):
            image = {
                "src": img.get('src', ''),
                "alt": img.get('alt', ''),
                "title": img.get('title', '')
            }
            images.append(image)
        
        return {
            "count": len(images),
            "with_alt": len([img for img in images if img["alt"]]),
            "with_title": len([img for img in images if img["title"]]),
            "images": images
        }
    
    def _extract_links(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract and analyze links."""
        links = []
        
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href:
                links.append({
                    "href": href,
                    "text": link.get_text().strip(),
                    "is_internal": href.startswith('/') or href.startswith('#')
                })
        
        return {
            "count": len(links),
            "internal": len([link for link in links if link["is_internal"]]),
            "external": len([link for link in links if not link["is_internal"]]),
            "links": links
        }
    
    async def _get_keyword_ranking(
        self,
        engine: str,
        keyword: str,
        domain: str
    ) -> Dict[str, Any]:
        """Get keyword ranking from a search engine."""
        # Simulate API call to search engine
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return {
            "position": 0,  # Would be actual position from search engine
            "volume": 0,    # Would be search volume from search engine
            "difficulty": 0 # Would be keyword difficulty from search engine
        }
    
    def _calculate_visibility_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate visibility score from SEO metrics."""
        score = 0.0
        
        # Title score
        if metrics["title"]["length"] > 0:
            score += min(metrics["title"]["length"] / 60, 1.0) * 0.3
        
        # Meta description score
        if metrics["meta_description"]["length"] > 0:
            score += min(metrics["meta_description"]["length"] / 160, 1.0) * 0.3
        
        # Headings score
        headings_count = sum(len(headings) for headings in metrics["headings"].values())
        score += min(headings_count / 10, 1.0) * 0.2
        
        # Links score
        score += min(metrics["links"]["count"] / 50, 1.0) * 0.2
        
        return score
    
    def _calculate_content_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate content quality score from SEO metrics."""
        score = 0.0
        
        # Content length score
        score += min(metrics["content_length"] / 2000, 1.0) * 0.4
        
        # Images score
        if metrics["images"]["count"] > 0:
            score += (
                metrics["images"]["with_alt"] / metrics["images"]["count"] * 0.3 +
                metrics["images"]["with_title"] / metrics["images"]["count"] * 0.3
            )
        
        return score
    
    def _calculate_user_experience_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate user experience score from SEO metrics."""
        score = 0.0
        
        # Load time score
        score += max(0, 1 - metrics["load_time"] / 3) * 0.4
        
        # Internal links score
        if metrics["links"]["count"] > 0:
            score += metrics["links"]["internal"] / metrics["links"]["count"] * 0.3
        
        # External links score
        if metrics["links"]["count"] > 0:
            score += min(metrics["links"]["external"] / 10, 1.0) * 0.3
        
        return score 