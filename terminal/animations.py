import sys
import time
import threading
import itertools
import random
from threading import Event

class Colors:
    SPINNER = '\033[38;2;129;161;193m'  # #81a1c1
    GREEN = '\033[92m'
    RESET = '\033[0m'

class Animations:
    """
    Generic terminal animation class.
    Can support multiple types of animations (spinner, progress bar, etc.)
    Currently implements a spinner.
    """

    def __init__(self, message="Processing...", animation_type="spinner"):
        self.message = message
        self.animation_type = animation_type
        self._stop_event = Event()
        self._thread = None
        self.gaming_phrases = [
            "Calculating epic loot drops...",
            "Respawning brain cells...",
            "Leveling up logic...",
            "Casting spell: Analyze...",
            "Looting the nearest thought chest...",
            "Unlocking hidden quest...",
            "Grinding XP for wisdom...",
            "Deploying tactical neurons...",
            "Loading epic decision matrix...",
            "Charging ultimate ability...",
            "Summoning the RNG gods...",
            "Scanning enemy strategies...",
            "Equipping thinking helmet...",
            "Buffing critical thinking...",
            "Hacking the mainframe of thought...",
            "Checking inventory of ideas...",
            "Teleporting to solution zone...",
            "Activating brain cooldown...",
            "Casting confusion resist...",
            "Executing combo of cognition..."
        ]

        # Define animation frames for spinner (can add more animations later)
        self.animations = {
            "spinner": ['|', '/', '-', '\\']
        }

    def _spinner_task(self):
        """Internal spinner animation loop"""
        for frame in itertools.cycle(self.animations["spinner"]):
            if self._stop_event.is_set():
                break
            sys.stdout.write(f'\r  {Colors.SPINNER}{frame}{Colors.RESET}  {self.message}')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write(f'\r{Colors.GREEN}✓{Colors.RESET}  {self.message} Done!     \n')

    def start(self, message=None, animation_type=None):
        """Start the animation in a separate thread"""
        if message:
            self.message = message
        if animation_type:
            self.animation_type = animation_type
        if self._thread and self._thread.is_alive():
            return  # already running

        self._stop_event.clear()
        if self.animation_type == "spinner":
            self._thread = threading.Thread(target=self._spinner_task)
        # future animation types can be added here
        self._thread.start()

    def stop(self):
        """Stop the animation"""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def run_with_animation(self, func, *args, message=None, animation_type=None, **kwargs):
        """
        Run any function while displaying the animation.
        Returns the result of the function.
        """

        self.start(message, animation_type)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            self.stop()

    def run_with_animation_but_random(self, func, *args, **kwargs):

        random_message = random.choice(self.gaming_phrases)

        self.start(random_message, None)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            self.stop()

