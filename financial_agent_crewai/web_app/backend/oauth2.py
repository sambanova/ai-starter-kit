import datetime
from typing import Any, Dict

import jwt
from fastapi.exceptions import HTTPException
from jwt.exceptions import InvalidTokenError

from financial_agent_crewai.web_app.backend import schemas
from financial_agent_crewai.web_app.backend.config import settings

SECRET_KEY = settings.secret_key
ALGORITHM = settings.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes


def create_access_token(data: Dict[str, Any]) -> Any:
    to_encode = data.copy()

    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({'exp': expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_access_token(token: str, credentials_exception: HTTPException) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        session_token: str = str(payload.get('session_token'))
        if session_token is None:
            raise credentials_exception
        token_data = schemas.TokenData(session_token=session_token)
    except InvalidTokenError:
        raise credentials_exception

    if token_data.session_token is not None:
        return token_data.session_token
    else:
        raise credentials_exception
