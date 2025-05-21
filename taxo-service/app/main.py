from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router as api_router

app = FastAPI(
    title="Taxo Service",
    description="EU Taxonomy alignment analysis service for ESG reports",
    version="1.0.0",
)

# Add CORS middleware for internal services only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific internal services
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True) 