#!/usr/bin/env bash
cd backend

set -o errexit

pip install -r requirements.txt

python manage.py collectstatic --no-input

python manage.py migrate