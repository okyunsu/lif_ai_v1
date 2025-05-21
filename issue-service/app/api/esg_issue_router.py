from fastapi import APIRouter, UploadFile, File
from app.domain.controller.esg_issue_controller import ESGIssueController

router = APIRouter(prefix="/esg/issues", tags=["ESG Issues"])

@router.post("/")
def extract_issues(file: UploadFile = File(...)):
    return ESGIssueController.extract_issues(file)