from pydantic import BaseModel, Field, HttpUrl


class Source(BaseModel):
    url: HttpUrl
    quote: str = Field(min_length=40, description="A verbatim passage from the fetched page that supports a claim.")


class Brief(BaseModel):
    title: str
    summary: str = Field(description="Three to five sentences. Every claim must be supported by a source quote.")
    sections: list[str]
    sources: list[Source] = Field(min_length=3)
