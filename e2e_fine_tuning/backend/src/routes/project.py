from fastapi import APIRouter, HTTPException, status

import e2e_fine_tuning.backend.src.schemas as schemas
from e2e_fine_tuning.backend.src.dependencies import get_byoc_service, get_hf_handler, get_sdk_service

router = APIRouter(prefix='/project', tags=['Project'])

sdk_service = get_sdk_service()
byoc_service = get_byoc_service()
hf_handler = get_hf_handler()


@router.get('/')
def get_projects() -> None:
    projects = sdk_service.get_projects()
    if projects is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'No projects found')
    return projects


@router.get('/{project_name}')
def get_project(project_name: str) -> schemas.Project:
    project_id = sdk_service.get_project(project_name)
    if project_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'No project with name: {project_name} found')
    return sdk_service.get_project(project_name)


@router.post('/')
def create_project(project_info: schemas.ProjectCreate) -> None:
    pass
