"""Aggregates versioned business endpoints: config, camera, events, outputs, system."""

from fastapi import APIRouter

from ..routers import camera_router, config_router, events_router, outputs_router, system_router

api_router = APIRouter()
api_router.include_router(config_router)
api_router.include_router(camera_router)
api_router.include_router(events_router)
api_router.include_router(outputs_router)
api_router.include_router(system_router)
