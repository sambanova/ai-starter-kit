from celery.app import Celery

from e2e_fine_tuning.backend.src.config import settings

redis_url = settings.redis_url
celery_app = Celery(__name__, broker=redis_url, backend=redis_url)

# celery_app.conf.update(
#     result_expires=3600,
# )
