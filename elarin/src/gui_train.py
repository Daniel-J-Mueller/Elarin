"""PyGame-based training interface with rating buttons."""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import pygame
import torch


class GUITrain(logging.Handler):
    """Display motor output and accept ratings using PyGame."""

    def __init__(
        self,
        motor,
        width: int = 224,
        height: int = 224,
        buffer_seconds: float = 30.0,
    ) -> None:
        super().__init__()
        pygame.init()
        self.motor = motor
        self.width = width
        self.height = height
        self.buffer_seconds = buffer_seconds
        self.top_h = 40
        self.right_w = 100
        self.input_h = 20
        total_w = width + self.right_w
        total_h = height + self.top_h + self.input_h
        self.screen = pygame.display.set_mode((total_w, total_h))
        pygame.display.set_caption("Elarin Training GUI")
        self.font = pygame.font.SysFont(None, 20)
        self.input_buffer = ""
        self.log: List[Tuple[float, str, int]] = []
        self.errors: List[str] = []
        self.context: Deque[Tuple[float, torch.Tensor]] = deque()
        self.rating_rects: List[Tuple[pygame.Rect, int]] = []
        self.running = True
        self.last_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.last_text = ""
        self.last_audio = 0.0

    # ``logging.Handler`` override
    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        token = getattr(record, "token_id", -1)
        if record.levelno >= logging.WARNING:
            self.errors.append(record.getMessage())
            self.errors = self.errors[-3:]
        else:
            self.log.append((record.created, record.getMessage(), token))
            self.log = self.log[-10:]

    def add_context(self, ctx: torch.Tensor) -> None:
        now = time.time()
        self.context.append((now, ctx.detach().cpu()))
        cutoff = now - self.buffer_seconds
        while self.context and self.context[0][0] < cutoff:
            self.context.popleft()

    def update(self, frame: np.ndarray, text: str = "", audio_level: float = 0.0) -> None:
        self.last_frame = frame
        self.last_text = text
        self.last_audio = audio_level
        self.draw()

    def draw(self) -> None:
        surface = pygame.surfarray.make_surface(self.last_frame.swapaxes(0, 1))
        if surface.get_width() != self.width or surface.get_height() != self.height:
            surface = pygame.transform.smoothscale(surface, (self.width, self.height))
        self.screen.fill((0, 0, 0))
        self.screen.blit(surface, (0, self.top_h))

        # top error box
        pygame.draw.rect(
            self.screen, (60, 0, 0), (0, 0, self.width + self.right_w, self.top_h)
        )
        y = 2
        for line in self.errors:
            surf = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (5, y))
            y += self.font.get_height()

        # rating buttons
        pygame.draw.rect(
            self.screen,
            (20, 20, 20),
            (self.width, self.top_h, self.right_w, self.height),
        )
        self.rating_rects.clear()
        for i, r in enumerate(range(-5, 6)):
            rect = pygame.Rect(
                self.width + 10, self.top_h + 10 + i * 20, self.right_w - 20, 18
            )
            pygame.draw.rect(self.screen, (70, 70, 70), rect)
            label = self.font.render(f"{r:+d}", True, (255, 255, 255))
            self.screen.blit(label, (rect.x + 5, rect.y))
            self.rating_rects.append((rect, r))

        # text overlay
        clean_text = self.last_text.strip()
        if clean_text:
            txt_surf = self.font.render(clean_text, True, (255, 255, 255))
            self.screen.blit(txt_surf, (5, self.top_h + 5))

        # input box
        pygame.draw.rect(
            self.screen,
            (20, 20, 20),
            (0, self.top_h + self.height, self.width + self.right_w, self.input_h),
        )
        if self.input_buffer:
            inp = self.font.render(self.input_buffer, True, (255, 255, 255))
            self.screen.blit(inp, (5, self.top_h + self.height + 2))

        pygame.display.flip()
        pygame.event.pump()

    def poll_text_input(self) -> Optional[str]:
        submitted: Optional[str] = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    submitted = self.input_buffer
                    self.input_buffer = ""
                elif event.key == pygame.K_BACKSPACE:
                    self.input_buffer = self.input_buffer[:-1]
                else:
                    if event.unicode and 32 <= ord(event.unicode) <= 126:
                        self.input_buffer += event.unicode
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                for rect, val in self.rating_rects:
                    if rect.collidepoint(x, y):
                        self.apply_rating(val)
        self.draw()
        return submitted

    def apply_rating(self, rating: int) -> None:
        if not self.log:
            return
        ts, msg, tok = self.log[-1]
        if tok == -1:
            return
        self.motor.reinforce_output(rating, tok)
        with Path("persistent/cli_feedback.log").open("a") as f:
            f.write(f"{time.time()}\t{tok}\t{rating}\t{msg}\n")

    def close(self) -> None:
        pygame.quit()
