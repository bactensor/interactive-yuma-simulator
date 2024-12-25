ARG BASE_IMAGE=python:3.11-slim


FROM $BASE_IMAGE AS base-image
LABEL builder=true

WORKDIR /root/src/

RUN pip3 install --no-cache-dir 'pdm>=2.12,<3'

COPY pyproject.toml pdm.lock ./
COPY ./yuma_simulation /root/src/yuma_simulation
RUN pdm lock --check

RUN apt-get update && apt-get install -y git


RUN pdm config python.use_venv False && pdm sync --prod --group :all
RUN mkdir -p /opt/ && mv __pypackages__/3.11/ /opt/pypackages/
ENV PATH=/opt/pypackages/bin:$PATH
ENV PYTHONPATH=/opt/pypackages/lib:$PYTHONPATH

COPY ./app/src/ ./app/envs/prod/entrypoint.sh ./app/envs/prod/gunicorn.conf.py ./app/envs/prod/celery-entrypoint.sh ./app/envs/prod/prometheus-cleanup.sh /root/src/
RUN chmod +x /root/src/entrypoint.sh \
             /root/src/prometheus-cleanup.sh \
             /root/src/manage.py

RUN python3 -m compileall -b -f -q /root/
RUN ENV=prod ENV_FILL_MISSING_VALUES=1 SECRET_KEY=dummy python3 manage.py collectstatic --no-input --clear


FROM $BASE_IMAGE AS secondary-image
LABEL builder=false

RUN apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /root/src/
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/pypackages/bin:$PATH
ENV PYTHONPATH=/opt/pypackages/lib:$PYTHONPATH

COPY --from=base-image /root/src/ /root/src/
COPY --from=base-image /opt/pypackages/ /opt/pypackages/



CMD ["./entrypoint.sh"]
