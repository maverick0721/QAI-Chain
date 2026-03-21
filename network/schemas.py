from __future__ import annotations

from pydantic import BaseModel, Field


class PeerRegistrationRequest(BaseModel):
    address: str = Field(min_length=1)


class PeerListResponse(BaseModel):
    peers: list[str]


class TransactionCreateRequest(BaseModel):
    receiver: str = Field(min_length=1)
    amount: float = Field(gt=0)


class TransactionReceiveRequest(BaseModel):
    sender: str = Field(min_length=1)
    receiver: str = Field(min_length=1)
    amount: float = Field(gt=0)
    signature: str | None = None


class TransactionCreatedResponse(BaseModel):
    status: str
    sender: str


class StatusResponse(BaseModel):
    status: str


class BlockResponse(BaseModel):
    status: str


class ChainBlockResponse(BaseModel):
    index: int
    hash: str
    tx: int


class BlockReceiveRequest(BaseModel):
    index: int
    transactions: list[TransactionReceiveRequest]
    previous_hash: str
    nonce: int
    hash: str
    zk_proof: str | None = None


class MineResponse(BaseModel):
    status: str
    hash: str | None = None
    zk_proof: str | None = None