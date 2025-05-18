"""Utility for visualizing what Elarin sees using PyGame."""

from typing import Optional
import numpy as np
import pygame


class Viewer:
    """Small window that displays the current frame with text and audio meter."""

    def __init__(self, width: int = 224, height: int = 224, bar_height: int = 30) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.bar_height = bar_height
        self.screen = pygame.display.set_mode((width, height + bar_height))
        pygame.display.set_caption("Elarin Viewer")
        self.font = pygame.font.SysFont(None, 24)

    def update(self, frame: np.ndarray, text: str = "", audio_level: float = 0.0) -> None:
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))
        pygame.draw.rect(
            self.screen, (0, 0, 0), (0, self.height, self.width, self.bar_height)
        )
        txt_surf = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(txt_surf, (5, self.height + 5))
        meter_h = self.bar_height - 10
        level = int(max(0.0, min(1.0, audio_level)) * meter_h)
        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (self.width - 15, self.height + meter_h - level + 5, 10, level),
        )
        pygame.display.flip()
        pygame.event.pump()

    def close(self) -> None:
        pygame.quit()
