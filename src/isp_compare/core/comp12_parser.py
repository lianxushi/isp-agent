"""
Comp12 RAW Format Parser
========================

Parses Comp12 RAW format images with the following specifications:
- Bit Depth: 12-bit
- Pixel Arrangement: Compact (no padding)
- Endianness: Little-endian (LSB)
- Supported Patterns: RGGB, BGGR, GRBG, GBRG
- Resolution: 3840x2160 or 1920x1080

Author: ISP Team
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Literal
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Comp12Config:
    """Comp12 RAW configuration"""
    width: int = 3840
    height: int = 2160
    pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"] = "RGGB"
    bit_depth: int = 12


class Comp12ParseError(Exception):
    """Exception raised when Comp12 parsing fails"""
    pass


class Comp12Parser:
    """
    Comp12 RAW format parser.
    
    Comp12 format characteristics:
    - 12-bit有效像素紧密排列，无填充位
    - 小端 (LSB) 存储
    - 无文件头
    - 转换为 Raw16 时，有效像素放置在低12位，高4位补0
    
    Example:
        >>> parser = Comp12Parser()
        >>> raw16 = parser.parse("input.raw", width=3840, height=2160, pattern="RGGB")
        >>> parser.save_for_cmodel(raw16, "output.raw")
    """
    
    SUPPORTED_PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]
    SUPPORTED_RESOLUTIONS = [(3840, 2160), (1920, 1080)]
    
    def __init__(self, config: Optional[Comp12Config] = None):
        """
        Initialize Comp12 parser.
        
        Args:
            config: Optional Comp12Config with default values
        """
        self.config = config or Comp12Config()
        self._validate_config()
        logger.info(f"Comp12Parser initialized: {self.config.width}x{self.config.height} {self.config.pattern}")
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if self.config.pattern not in self.SUPPORTED_PATTERNS:
            raise Comp12ParseError(
                f"Unsupported pattern: {self.config.pattern}. "
                f"Supported: {self.SUPPORTED_PATTERNS}"
            )
        
        if (self.config.width, self.config.height) not in self.SUPPORTED_RESOLUTIONS:
            raise Comp12ParseError(
                f"Unsupported resolution: {self.config.width}x{self.config.height}. "
                f"Supported: {self.SUPPORTED_RESOLUTIONS}"
            )
        
        if not 1 <= self.config.bit_depth <= 16:
            raise Comp12ParseError(
                f"Invalid bit depth: {self.config.bit_depth}"
            )
    
    def parse(self, filepath: str) -> np.ndarray:
        """
        Parse Comp12 RAW file to 16-bit RAW image.
        
        Args:
            filepath: Path to Comp12 RAW file
            
        Returns:
            numpy.ndarray: 16-bit RAW image (H x W)
            
        Raises:
            Comp12ParseError: If parsing fails
            FileNotFoundError: If file does not exist
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Comp12 file not found: {filepath}")
        
        try:
            # 1. Read 12-bit data as 16-bit (little-endian)
            raw12 = np.fromfile(str(filepath), dtype=np.uint16)
            
            # 2. Validate size
            expected_pixels = self.config.width * self.config.height
            if raw12.size < expected_pixels:
                raise Comp12ParseError(
                    f"File too small: {raw12.size} pixels, "
                    f"expected {expected_pixels} for {self.config.width}x{self.config.height}"
                )
            
            # 3. Take only expected pixels (in case file has extra data)
            raw12 = raw12[:expected_pixels]
            
            # 4. Already 16-bit, valid pixels in lower 12 bits
            raw16 = raw12.astype(np.uint16)
            
            # 5. Reshape to 2D image
            raw16 = raw16.reshape((self.config.height, self.config.width))
            
            logger.debug(f"Parsed Comp12: {filepath.name}, shape={raw16.shape}, "
                        f"dtype={raw16.dtype}, range=[{raw16.min()}, {raw16.max()}]")
            
            return raw16
            
        except Comp12ParseError:
            raise
        except Exception as e:
            raise Comp12ParseError(f"Failed to parse Comp12 file {filepath}: {e}")
    
    def save_for_cmodel(self, raw16: np.ndarray, output_path: str) -> None:
        """
        Save 16-bit RAW image for CModel input.
        
        Args:
            raw16: 16-bit RAW image
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as raw 16-bit little-endian
        raw16.astype(np.uint16).tofile(str(output_path))
        logger.debug(f"Saved RAW for CModel: {output_path}")
    
    def get_bayer_channel_map(self) -> dict:
        """
        Get Bayer pattern channel mapping.
        
        Returns:
            dict: Mapping of (row, col) to channel name
        """
        pattern = self.config.pattern
        channel_map = {}
        
        for r in range(2):
            for c in range(2):
                if pattern == "RGGB":
                    ch = "R" if (r, c) == (0, 0) else "Gr" if (r, c) == (0, 1) else "Gb" if (r, c) == (1, 0) else "B"
                elif pattern == "BGGR":
                    ch = "B" if (r, c) == (0, 0) else "Gb" if (r, c) == (0, 1) else "Gr" if (r, c) == (1, 0) else "R"
                elif pattern == "GRBG":
                    ch = "Gr" if (r, c) == (0, 0) else "R" if (r, c) == (0, 1) else "B" if (r, c) == (1, 0) else "Gb"
                elif pattern == "GBRG":
                    ch = "Gb" if (r, c) == (0, 0) else "B" if (r, c) == (0, 1) else "R" if (r, c) == (1, 0) else "Gr"
                channel_map[(r, c)] = ch
        
        return channel_map
    
    @staticmethod
    def validate_file(filepath: str, expected_pixels: int) -> bool:
        """
        Validate Comp12 file size.
        
        Args:
            filepath: Path to file
            expected_pixels: Expected number of pixels
            
        Returns:
            bool: True if valid
        """
        try:
            size = os.path.getsize(filepath)
            # 2 bytes per pixel for 16-bit
            actual_pixels = size // 2
            return actual_pixels >= expected_pixels
        except Exception:
            return False
