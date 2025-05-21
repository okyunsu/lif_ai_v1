import uuid
import shutil
from fastapi import UploadFile, HTTPException
from app.domain.service.esg_issue_service import (
    extract_esg_issues_from_pdf,
    save_issues_to_json,
    save_issues_to_csv
)
import os

class ESGIssueController:
    @staticmethod
    def extract_issues(file: UploadFile):
        try:
            temp_path = f"/tmp/{file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            issues = extract_esg_issues_from_pdf(temp_path)
            os.remove(temp_path)

            issue_id = str(uuid.uuid4())[:8]
            json_path = save_issues_to_json(issues, f"esg_issues_{issue_id}.json")
            csv_path = save_issues_to_csv(issues, f"esg_issues_{issue_id}.csv")

            return {
                "issue_count": len(issues),
                "json_path": json_path,
                "csv_path": csv_path,
                "issues": issues
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"이슈 추출 실패: {str(e)}")
