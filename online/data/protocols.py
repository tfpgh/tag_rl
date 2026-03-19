from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandSegment:
    duration_s: float
    linear: float
    angular: float
    label: str


@dataclass(frozen=True, slots=True)
class ScriptedProtocol:
    name: str
    segments: tuple[CommandSegment, ...]


def default_protocols() -> tuple[ScriptedProtocol, ...]:
    return (
        ScriptedProtocol(
            name="straight_fwd",
            segments=(
                CommandSegment(1.0, 0.0, 0.0, "settle"),
                CommandSegment(2.0, 0.45, 0.0, "drive"),
                CommandSegment(1.0, 0.0, 0.0, "stop"),
            ),
        ),
        ScriptedProtocol(
            name="straight_rev",
            segments=(
                CommandSegment(1.0, 0.0, 0.0, "settle"),
                CommandSegment(2.0, -0.45, 0.0, "drive"),
                CommandSegment(1.0, 0.0, 0.0, "stop"),
            ),
        ),
        ScriptedProtocol(
            name="turn_left_in_place",
            segments=(
                CommandSegment(1.0, 0.0, 0.0, "settle"),
                CommandSegment(2.0, 0.0, 0.35, "turn"),
                CommandSegment(1.0, 0.0, 0.0, "stop"),
            ),
        ),
        ScriptedProtocol(
            name="turn_right_in_place",
            segments=(
                CommandSegment(1.0, 0.0, 0.0, "settle"),
                CommandSegment(2.0, 0.0, -0.35, "turn"),
                CommandSegment(1.0, 0.0, 0.0, "stop"),
            ),
        ),
        ScriptedProtocol(
            name="arc_left",
            segments=(
                CommandSegment(1.0, 0.0, 0.0, "settle"),
                CommandSegment(2.5, 0.4, 0.16, "arc"),
                CommandSegment(1.0, 0.0, 0.0, "stop"),
            ),
        ),
        ScriptedProtocol(
            name="arc_right",
            segments=(
                CommandSegment(1.0, 0.0, 0.0, "settle"),
                CommandSegment(2.5, 0.4, -0.16, "arc"),
                CommandSegment(1.0, 0.0, 0.0, "stop"),
            ),
        ),
        ScriptedProtocol(
            name="pulse_forward",
            segments=(
                CommandSegment(1.0, 0.0, 0.0, "settle"),
                CommandSegment(0.5, 0.5, 0.0, "pulse1"),
                CommandSegment(0.5, 0.0, 0.0, "coast1"),
                CommandSegment(0.5, 0.5, 0.0, "pulse2"),
                CommandSegment(1.0, 0.0, 0.0, "stop"),
            ),
        ),
        ScriptedProtocol(
            name="pulse_turn",
            segments=(
                CommandSegment(1.0, 0.0, 0.0, "settle"),
                CommandSegment(0.5, 0.0, 0.4, "pulse1"),
                CommandSegment(0.5, 0.0, 0.0, "coast1"),
                CommandSegment(0.5, 0.0, -0.4, "pulse2"),
                CommandSegment(1.0, 0.0, 0.0, "stop"),
            ),
        ),
        ScriptedProtocol(
            name="asymmetry_test",
            segments=(
                CommandSegment(1.0, 0.0, 0.0, "settle"),
                CommandSegment(1.5, 0.35, 0.08, "bias_left"),
                CommandSegment(1.0, 0.0, 0.0, "stop1"),
                CommandSegment(1.5, 0.35, -0.08, "bias_right"),
                CommandSegment(1.0, 0.0, 0.0, "stop2"),
            ),
        ),
    )
