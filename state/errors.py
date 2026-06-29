#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""业务异常。"""


class TubeGrabberError(Exception):
    pass


class SlotEmptyError(TubeGrabberError):
    pass


class SlotOccupiedError(TubeGrabberError):
    pass


class SlotUnknownError(TubeGrabberError):
    pass


class CoordMissingError(TubeGrabberError):
    pass


class SnapshotMissingError(TubeGrabberError):
    pass


class SafetyViolation(TubeGrabberError):
    pass
