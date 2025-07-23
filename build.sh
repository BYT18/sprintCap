#!/usr/bin/env bash
cd src/backend

set -o errexit

pip install -r requirements.txt

python manage.py collectstatic --no-input

python manage.py migrate