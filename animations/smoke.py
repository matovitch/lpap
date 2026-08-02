"""Tiny Manim smoke scene to validate the animations/ Pixi workflow."""

from manim import BLUE, DOWN, Scene, Text, Write


class SmokeTest(Scene):
    """Dump clip: title + subtitle. Not part of the LPAP storyboard."""

    def construct(self) -> None:
        title = Text("LPAP Manim smoke test", font_size=42)
        subtitle = Text("animations/ workflow ok", font_size=28, color=BLUE)
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(Write(title))
        self.play(Write(subtitle))
        self.wait(1.0)
