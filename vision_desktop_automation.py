"""
Vision-Based Desktop Automation for Multi-Bot Application
==========================================================

This module provides advanced desktop automation using computer vision and AI
to interact with Reddit (and other forums) through the browser UI like a real human.

Features:
- Screenshot analysis and OCR
- AI vision (GPT-4 Vision / Claude Vision) for UI understanding
- Human-like mouse movements and clicks
- Realistic keyboard typing with variations
- Template matching for UI element detection
- Adaptive behavior based on visual feedback
- Anti-detection measures

Author: AI Assistant
Date: 2025
"""

import asyncio
import base64
import io
import random
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

# Desktop automation
import pyautogui
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key

# Computer vision
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import easyocr

# Screenshots
from mss import mss

# AI Vision
import openai
from anthropic import Anthropic

# Image processing
from scipy import interpolate
import imagehash

# Utilities
import os
from dotenv import load_dotenv

# Import from main app
from multi_bot_forum_app import Bot, logger, Config

load_dotenv()

# Configure PyAutoGUI safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5


# ============================================================================
# HUMAN-LIKE MOUSE MOVEMENT
# ============================================================================

class HumanMouseController:
    """Simulates human-like mouse movements using Bezier curves"""

    def __init__(self):
        self.mouse = MouseController()
        self.current_pos = pyautogui.position()

    def get_current_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        return pyautogui.position()

    def generate_bezier_curve(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        num_points: int = 50,
        distortion: float = 0.2
    ) -> List[Tuple[int, int]]:
        """
        Generate a Bezier curve path from start to end with random distortion

        Args:
            start: Starting coordinates (x, y)
            end: Ending coordinates (x, y)
            num_points: Number of points in the curve
            distortion: Amount of random curve (0.0 to 1.0)
        """
        # Calculate control points with randomness
        distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        num_control_points = 2

        control_points = [start]

        for i in range(num_control_points):
            t = (i + 1) / (num_control_points + 1)
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])

            # Add random distortion perpendicular to the direct line
            offset = distance * distortion * random.uniform(-1, 1)
            angle = np.arctan2(end[1] - start[1], end[0] - start[0]) + np.pi / 2

            x += offset * np.cos(angle)
            y += offset * np.sin(angle)

            control_points.append((x, y))

        control_points.append(end)

        # Generate Bezier curve
        control_points = np.array(control_points)
        t = np.linspace(0, 1, num_points)

        # Calculate Bezier curve points
        n = len(control_points) - 1
        curve_points = []

        for t_val in t:
            point = np.zeros(2)
            for i, cp in enumerate(control_points):
                # Bernstein polynomial
                bernstein = (
                    np.math.comb(n, i) *
                    (t_val ** i) *
                    ((1 - t_val) ** (n - i))
                )
                point += bernstein * cp
            curve_points.append((int(point[0]), int(point[1])))

        return curve_points

    async def move_to(
        self,
        x: int,
        y: int,
        duration: Optional[float] = None,
        human_like: bool = True
    ):
        """
        Move mouse to position with human-like movement

        Args:
            x: Target x coordinate
            y: Target y coordinate
            duration: Time to complete movement (random if None)
            human_like: Use Bezier curve for natural movement
        """
        start = self.get_current_position()
        distance = np.sqrt((x - start[0])**2 + (y - start[1])**2)

        if duration is None:
            # Human reaction time + movement time based on distance
            duration = random.uniform(0.3, 0.7) + (distance / 1000)

        if human_like and distance > 50:
            # Use Bezier curve for longer movements
            curve_points = self.generate_bezier_curve(
                start, (x, y),
                num_points=max(20, int(distance / 10)),
                distortion=random.uniform(0.1, 0.3)
            )

            # Move along the curve
            time_per_point = duration / len(curve_points)

            for point in curve_points:
                pyautogui.moveTo(point[0], point[1], duration=0)
                await asyncio.sleep(time_per_point * random.uniform(0.8, 1.2))

            # Ensure we end exactly at target
            pyautogui.moveTo(x, y, duration=0)

        else:
            # Direct movement for short distances
            pyautogui.moveTo(x, y, duration=duration)
            await asyncio.sleep(duration)

        # Small random movement at end (human jitter)
        if random.random() < 0.3:
            offset_x = random.randint(-2, 2)
            offset_y = random.randint(-2, 2)
            pyautogui.moveRel(offset_x, offset_y, duration=0.1)
            await asyncio.sleep(0.1)

    async def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = 'left',
        clicks: int = 1
    ):
        """
        Click at position with human-like behavior

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: 'left' or 'right'
            clicks: Number of clicks
        """
        if x is not None and y is not None:
            await self.move_to(x, y)

        # Small delay before click (human reaction time)
        await asyncio.sleep(random.uniform(0.05, 0.15))

        # Click with slight randomness in timing
        for _ in range(clicks):
            pyautogui.click(button=button)
            if clicks > 1:
                await asyncio.sleep(random.uniform(0.1, 0.2))

        # Tiny movement after click (natural hand movement)
        if random.random() < 0.4:
            pyautogui.moveRel(
                random.randint(-1, 1),
                random.randint(-1, 1),
                duration=0.05
            )

    async def scroll(self, amount: int, direction: str = 'down'):
        """
        Scroll with human-like behavior

        Args:
            amount: Scroll amount
            direction: 'up' or 'down'
        """
        scroll_amount = amount if direction == 'down' else -amount

        # Break into smaller scrolls (humans don't scroll in one motion)
        chunks = random.randint(2, 5)
        chunk_size = scroll_amount // chunks

        for i in range(chunks):
            # Varying scroll speeds
            variation = random.uniform(0.8, 1.2)
            pyautogui.scroll(int(chunk_size * variation))

            # Small pause between scrolls
            await asyncio.sleep(random.uniform(0.1, 0.3))

        # Occasional small scroll back (human correction)
        if random.random() < 0.2:
            await asyncio.sleep(random.uniform(0.2, 0.5))
            pyautogui.scroll(int(chunk_size * 0.1 * -1))


# ============================================================================
# HUMAN-LIKE KEYBOARD TYPING
# ============================================================================

class HumanKeyboardController:
    """Simulates human-like keyboard typing"""

    def __init__(self):
        self.keyboard = KeyboardController()

    async def type_text(
        self,
        text: str,
        wpm: Optional[int] = None,
        error_rate: float = 0.02
    ):
        """
        Type text with human-like characteristics

        Args:
            text: Text to type
            wpm: Words per minute (random 40-80 if None)
            error_rate: Probability of making a typo (0.0 to 1.0)
        """
        if wpm is None:
            wpm = random.randint(40, 80)

        # Calculate base delay between characters
        chars_per_second = (wpm * 5) / 60  # Average 5 chars per word
        base_delay = 1.0 / chars_per_second

        i = 0
        while i < len(text):
            char = text[i]

            # Introduce occasional typos
            if random.random() < error_rate and char.isalnum():
                # Type wrong character
                wrong_char = self._get_nearby_key(char)
                pyautogui.write(wrong_char, interval=0)

                # Human reaction time to notice error
                await asyncio.sleep(random.uniform(0.2, 0.5))

                # Backspace to correct
                pyautogui.press('backspace')
                await asyncio.sleep(random.uniform(0.1, 0.2))

            # Type the correct character
            pyautogui.write(char, interval=0)

            # Variable delay between characters
            delay = base_delay * random.uniform(0.7, 1.5)

            # Longer delays after punctuation or spaces
            if char in '.,!?':
                delay *= random.uniform(1.5, 2.5)
            elif char == ' ':
                delay *= random.uniform(1.2, 1.8)
            elif char == '\n':
                delay *= random.uniform(2.0, 3.0)

            # Occasional thinking pauses
            if random.random() < 0.05:
                delay += random.uniform(0.5, 2.0)

            await asyncio.sleep(delay)
            i += 1

    def _get_nearby_key(self, char: str) -> str:
        """Get a nearby key on QWERTY keyboard for typo simulation"""
        keyboard_layout = {
            'q': ['w', 'a'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'],
            'r': ['e', 't', 'f'], 't': ['r', 'y', 'g'], 'y': ['t', 'u', 'h'],
            'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'], 'o': ['i', 'p', 'l'],
            'p': ['o', 'l'],
            'a': ['q', 's', 'z'], 's': ['a', 'w', 'd', 'x'], 'd': ['s', 'e', 'f', 'c'],
            'f': ['d', 'r', 'g', 'v'], 'g': ['f', 't', 'h', 'b'], 'h': ['g', 'y', 'j', 'n'],
            'j': ['h', 'u', 'k', 'm'], 'k': ['j', 'i', 'l'], 'l': ['k', 'o', 'p'],
            'z': ['a', 'x'], 'x': ['z', 's', 'c'], 'c': ['x', 'd', 'v'],
            'v': ['c', 'f', 'b'], 'b': ['v', 'g', 'n'], 'n': ['b', 'h', 'm'],
            'm': ['n', 'j']
        }

        char_lower = char.lower()
        if char_lower in keyboard_layout:
            nearby = random.choice(keyboard_layout[char_lower])
            return nearby.upper() if char.isupper() else nearby

        return char


# ============================================================================
# SCREENSHOT AND VISION ANALYSIS
# ============================================================================

@dataclass
class UIElement:
    """Represents a detected UI element"""
    name: str
    location: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    text: Optional[str] = None
    center: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.center is None:
            x, y, w, h = self.location
            self.center = (x + w // 2, y + h // 2)


class VisionAnalyzer:
    """Analyzes screenshots using OCR and AI vision models"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ocr_reader = easyocr.Reader(['en'])

        # Initialize AI vision clients
        self.openai_client = None
        self.anthropic_client = None

        if config.get('openai', {}).get('enabled'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.openai_client = openai

        if config.get('anthropic', {}).get('enabled'):
            self.anthropic_client = Anthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )

    def capture_screenshot(
        self,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Capture screenshot of screen or region

        Args:
            region: (x, y, width, height) or None for full screen
        """
        with mss() as sct:
            if region:
                monitor = {
                    'left': region[0],
                    'top': region[1],
                    'width': region[2],
                    'height': region[3]
                }
            else:
                monitor = sct.monitors[1]  # Primary monitor

            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            # Convert BGRA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        return img

    def find_text_on_screen(
        self,
        text: str,
        screenshot: Optional[np.ndarray] = None,
        case_sensitive: bool = False
    ) -> List[UIElement]:
        """
        Find text on screen using OCR

        Args:
            text: Text to find
            screenshot: Screenshot to search (captures new if None)
            case_sensitive: Case-sensitive search

        Returns:
            List of UIElement objects with locations
        """
        if screenshot is None:
            screenshot = self.capture_screenshot()

        # Use EasyOCR for better accuracy
        results = self.ocr_reader.readtext(screenshot)

        matches = []
        search_text = text if case_sensitive else text.lower()

        for (bbox, detected_text, confidence) in results:
            compare_text = detected_text if case_sensitive else detected_text.lower()

            if search_text in compare_text:
                # Convert bbox to x, y, width, height
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]

                x = int(min(x_coords))
                y = int(min(y_coords))
                width = int(max(x_coords) - x)
                height = int(max(y_coords) - y)

                element = UIElement(
                    name=f"text_{detected_text}",
                    location=(x, y, width, height),
                    confidence=confidence,
                    text=detected_text
                )
                matches.append(element)

        logger.info(f"Found {len(matches)} matches for text '{text}'")
        return matches

    def find_image_on_screen(
        self,
        template_path: str,
        screenshot: Optional[np.ndarray] = None,
        threshold: float = 0.8
    ) -> List[UIElement]:
        """
        Find image template on screen using template matching

        Args:
            template_path: Path to template image
            screenshot: Screenshot to search (captures new if None)
            threshold: Matching threshold (0.0 to 1.0)
        """
        if screenshot is None:
            screenshot = self.capture_screenshot()

        # Load template
        template = cv2.imread(template_path)
        if template is None:
            logger.error(f"Could not load template: {template_path}")
            return []

        # Convert to grayscale
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Template matching
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # Find locations above threshold
        locations = np.where(result >= threshold)
        matches = []

        h, w = template_gray.shape

        for pt in zip(*locations[::-1]):
            element = UIElement(
                name=f"image_{Path(template_path).stem}",
                location=(pt[0], pt[1], w, h),
                confidence=float(result[pt[1], pt[0]])
            )
            matches.append(element)

        # Remove overlapping matches (non-maximum suppression)
        matches = self._non_max_suppression(matches)

        logger.info(f"Found {len(matches)} matches for template {template_path}")
        return matches

    def _non_max_suppression(
        self,
        elements: List[UIElement],
        overlap_threshold: float = 0.5
    ) -> List[UIElement]:
        """Remove overlapping detections, keeping highest confidence"""
        if not elements:
            return []

        # Sort by confidence
        elements = sorted(elements, key=lambda x: x.confidence, reverse=True)

        keep = []

        while elements:
            best = elements.pop(0)
            keep.append(best)

            # Remove overlapping elements
            elements = [
                elem for elem in elements
                if self._calculate_iou(best.location, elem.location) < overlap_threshold
            ]

        return keep

    def _calculate_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union for two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    async def analyze_with_ai_vision(
        self,
        screenshot: np.ndarray,
        prompt: str,
        model: str = 'gpt-4-vision'
    ) -> str:
        """
        Analyze screenshot using AI vision model

        Args:
            screenshot: Screenshot to analyze
            prompt: Question/task for the AI
            model: 'gpt-4-vision' or 'claude-vision'

        Returns:
            AI response describing what it sees
        """
        # Convert screenshot to base64
        image_pil = Image.fromarray(screenshot)
        buffer = io.BytesIO()
        image_pil.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        if model == 'gpt-4-vision' and self.openai_client:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            return response.choices[0].message.content

        elif model == 'claude-vision' and self.anthropic_client:
            message = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            return message.content[0].text

        else:
            logger.error(f"AI vision model {model} not available")
            return ""

    def save_screenshot_with_annotations(
        self,
        screenshot: np.ndarray,
        elements: List[UIElement],
        output_path: str
    ):
        """Save screenshot with bounding boxes around detected elements"""
        img = Image.fromarray(screenshot)
        draw = ImageDraw.Draw(img)

        for element in elements:
            x, y, w, h = element.location
            # Draw rectangle
            draw.rectangle(
                [(x, y), (x + w, y + h)],
                outline='red',
                width=2
            )
            # Draw label
            if element.text:
                draw.text((x, y - 15), element.text, fill='red')

        img.save(output_path)
        logger.info(f"Saved annotated screenshot to {output_path}")


# ============================================================================
# VISION-BASED REDDIT AUTOMATION
# ============================================================================

class VisionBasedRedditBot:
    """
    Reddit bot that uses computer vision and AI to interact with browser
    like a real human user
    """

    def __init__(self, bot: Bot, config: Dict[str, Any]):
        self.bot = bot
        self.config = config

        self.mouse = HumanMouseController()
        self.keyboard = HumanKeyboardController()
        self.vision = VisionAnalyzer(config)

        self.browser_region: Optional[Tuple[int, int, int, int]] = None

    async def initialize_browser(self):
        """Open browser and navigate to Reddit"""
        logger.info(f"Initializing browser for bot {self.bot.name}")

        # Open browser (using default browser)
        pyautogui.hotkey('win', 'r')  # Windows Run dialog
        await asyncio.sleep(0.5)
        await self.keyboard.type_text('chrome.exe', wpm=60)
        await asyncio.sleep(0.3)
        pyautogui.press('enter')
        await asyncio.sleep(3)

        # Navigate to Reddit
        await self.keyboard.type_text('reddit.com', wpm=70)
        pyautogui.press('enter')
        await asyncio.sleep(5)

        # Detect browser window region (for faster screenshots)
        self.browser_region = None  # Use full screen for now

    async def login_to_reddit(self) -> bool:
        """Login to Reddit using vision-based automation"""
        logger.info(f"Logging in to Reddit as {self.bot.credentials['username']}")

        try:
            # Take screenshot
            screenshot = self.vision.capture_screenshot()

            # Look for "Log In" button
            login_elements = self.vision.find_text_on_screen("Log In", screenshot)

            if not login_elements:
                logger.info("Already logged in or login button not found")
                return True

            # Click on login button
            login_button = login_elements[0]
            await self.mouse.click(*login_button.center)
            await asyncio.sleep(2)

            # Wait for login form to appear
            screenshot = self.vision.capture_screenshot()

            # Find username field using AI vision
            prompt = "Where is the username input field? Provide coordinates."
            ai_response = await self.vision.analyze_with_ai_vision(
                screenshot,
                "Find the username or email input field on this Reddit login page. Describe its location."
            )
            logger.info(f"AI Vision: {ai_response}")

            # For now, use OCR to find "Username" text
            username_labels = self.vision.find_text_on_screen("Username", screenshot)

            if username_labels:
                # Click below the label (where input field likely is)
                label_center = username_labels[0].center
                await self.mouse.click(label_center[0], label_center[1] + 30)
                await asyncio.sleep(0.5)

                # Type username
                await self.keyboard.type_text(
                    self.bot.credentials['username'],
                    wpm=random.randint(50, 70),
                    error_rate=0.01
                )

            await asyncio.sleep(0.5)

            # Tab to password field
            pyautogui.press('tab')
            await asyncio.sleep(0.3)

            # Type password
            await self.keyboard.type_text(
                self.bot.credentials['password'],
                wpm=random.randint(50, 70),
                error_rate=0.01
            )

            await asyncio.sleep(0.5)

            # Press Enter or click login button
            pyautogui.press('enter')
            await asyncio.sleep(5)

            # Verify login
            screenshot = self.vision.capture_screenshot()
            if self.vision.find_text_on_screen(self.bot.credentials['username'], screenshot):
                logger.info(f"Successfully logged in as {self.bot.credentials['username']}")
                return True
            else:
                logger.error("Login verification failed")
                return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    async def navigate_to_subreddit(self, subreddit: str):
        """Navigate to a subreddit using vision"""
        logger.info(f"Navigating to r/{subreddit}")

        # Click address bar
        pyautogui.hotkey('ctrl', 'l')
        await asyncio.sleep(0.3)

        # Type subreddit URL
        await self.keyboard.type_text(
            f"reddit.com/r/{subreddit}",
            wpm=random.randint(60, 80)
        )
        pyautogui.press('enter')
        await asyncio.sleep(3)

    async def read_thread_and_post_comment(
        self,
        thread_title: str,
        comment_text: str
    ) -> bool:
        """
        Find a thread by title, read it, and post a comment using vision

        Args:
            thread_title: Title of thread to find
            comment_text: Comment to post
        """
        try:
            # Take screenshot
            screenshot = self.vision.capture_screenshot()

            # Find thread title
            thread_elements = self.vision.find_text_on_screen(
                thread_title[:30],  # Use first 30 chars
                screenshot
            )

            if not thread_elements:
                logger.warning(f"Thread '{thread_title}' not found on screen")
                return False

            # Click on thread
            await self.mouse.click(*thread_elements[0].center)
            await asyncio.sleep(3)

            # Scroll through thread (simulate reading)
            logger.info("Reading thread content...")
            for _ in range(random.randint(2, 4)):
                await self.mouse.scroll(random.randint(300, 500), 'down')
                await asyncio.sleep(random.uniform(1.5, 3.0))

            # Find comment box
            screenshot = self.vision.capture_screenshot()

            # Use AI vision to find comment box
            ai_response = await self.vision.analyze_with_ai_vision(
                screenshot,
                "Where is the comment input box on this Reddit post? Describe its location on the screen."
            )
            logger.info(f"AI Vision for comment box: {ai_response}")

            # Look for "Comment" text or placeholder
            comment_elements = self.vision.find_text_on_screen("Comment", screenshot)

            if comment_elements:
                # Click on comment box
                await self.mouse.click(*comment_elements[0].center)
                await asyncio.sleep(1)

                # Type comment with human-like behavior
                logger.info("Typing comment...")
                await self.keyboard.type_text(
                    comment_text,
                    wpm=random.randint(40, 60),
                    error_rate=0.02
                )

                await asyncio.sleep(1)

                # Find and click "Comment" button
                screenshot = self.vision.capture_screenshot()
                submit_buttons = self.vision.find_text_on_screen("Comment", screenshot)

                # Find the submit button (usually different from text box)
                for button in submit_buttons:
                    if button.center[1] > comment_elements[0].center[1]:
                        await self.mouse.click(*button.center)
                        break

                await asyncio.sleep(3)

                logger.info(f"Posted comment on thread '{thread_title}'")
                return True

            else:
                logger.error("Could not find comment box")
                return False

        except Exception as e:
            logger.error(f"Error posting comment: {e}")
            return False

    async def browse_like_human(self, duration_minutes: int = 5):
        """
        Browse Reddit naturally for a duration

        Args:
            duration_minutes: How long to browse
        """
        logger.info(f"Browsing Reddit naturally for {duration_minutes} minutes")

        end_time = time.time() + (duration_minutes * 60)

        while time.time() < end_time:
            # Scroll randomly
            scroll_amount = random.randint(200, 600)
            await self.mouse.scroll(scroll_amount, 'down')

            # Pause to "read"
            await asyncio.sleep(random.uniform(2, 8))

            # Occasionally click on a post
            if random.random() < 0.3:
                screenshot = self.vision.capture_screenshot()

                # Get all text elements
                elements = self.vision.ocr_reader.readtext(screenshot)

                if elements and len(elements) > 5:
                    # Click on a random post title
                    random_element = random.choice(elements[5:15])
                    bbox = random_element[0]
                    center_x = int((bbox[0][0] + bbox[2][0]) / 2)
                    center_y = int((bbox[0][1] + bbox[2][1]) / 2)

                    await self.mouse.click(center_x, center_y)
                    await asyncio.sleep(random.uniform(3, 7))

                    # Read post
                    for _ in range(random.randint(1, 3)):
                        await self.mouse.scroll(random.randint(200, 400), 'down')
                        await asyncio.sleep(random.uniform(2, 5))

                    # Go back
                    pyautogui.hotkey('alt', 'left')
                    await asyncio.sleep(2)

        logger.info("Finished browsing session")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_vision_automation():
    """Example of using vision-based automation"""

    from multi_bot_forum_app import Personality, Objective, ObjectiveType, BotStatus

    # Create example bot
    bot = Bot(
        id="vision_bot_1",
        name="VisionRedditBot",
        personality=Personality(
            id="pers_1",
            name="Friendly Helper",
            description="Friendly and helpful",
            tone="friendly",
            formality=0.4,
            enthusiasm=0.7,
            technical_level=0.5,
            verbosity=0.5,
            empathy=0.8,
            vocabulary_style="casual",
            response_patterns=["Hey!", "Great question!"]
        ),
        objectives=[],
        status=BotStatus.ACTIVE,
        credentials={
            'username': os.getenv('REDDIT_BOT1_USERNAME', 'demo_user'),
            'password': os.getenv('REDDIT_BOT1_PASSWORD', 'demo_pass')
        },
        metadata={},
        created_at=datetime.now(),
        last_active=datetime.now()
    )

    # Configuration
    config = {
        'openai': {'enabled': True},
        'anthropic': {'enabled': False}
    }

    # Create vision-based bot
    vision_bot = VisionBasedRedditBot(bot, config)

    # Initialize browser
    await vision_bot.initialize_browser()

    # Login
    success = await vision_bot.login_to_reddit()

    if success:
        # Navigate to subreddit
        await vision_bot.navigate_to_subreddit('learnpython')

        # Browse naturally
        await vision_bot.browse_like_human(duration_minutes=2)

        # Post a comment (example - you'd find a specific thread)
        # await vision_bot.read_thread_and_post_comment(
        #     "How do I learn Python?",
        #     "Great question! I'd recommend starting with the official Python tutorial."
        # )


if __name__ == "__main__":
    asyncio.run(example_vision_automation())
