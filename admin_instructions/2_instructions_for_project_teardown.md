# Cleanup Instructions

Execute the following steps once you are done with the lab series,
**and you have pushed your commits to GitHub for grading**, and, for the
Hugging Face Spaces / Streamlit Community Cloud / Render deployments, once
your domain expert has finished validating your model:

***Note:** You can also wait until the end of the semester after the marks have
been officially released to implement the cleanup in all the labs.*

## Local Docker Resources (Reverse Proxy Lab)

Shutdown the Docker containers and remove the images, networks, and volumes
created during the lab.

```shell
docker compose down -v
```

**Important:** if you started the stack with additional compose files
layered on top of the base file (for example, `docker compose -f docker-compose.yaml -f docker-compose-dev.yaml up`), tear down using that **exact same
combination** of `-f` flags:

```shell
docker compose -f docker-compose.yaml -f docker-compose-dev.yaml down -v
```

Running plain `docker compose down -v` only reads `docker-compose.yaml` by
default, and can silently miss containers, networks, or volumes that exist
only in an overlay file (`docker-compose-dev.yaml` or `docker-compose-prod.yaml`).

### Images to Delete

```shell
docker image rm servingmlmodels-flask-gunicorn-app servingmlmodels-nginx customized-ubuntu-server-smm
```

- `servingmlmodels-flask-gunicorn-app`
- `servingmlmodels-nginx`
- `customized-ubuntu-server-smm`

(Confirm the exact names with `docker images` first -- Compose derives an
image name from `<project-name>-<service-name>` unless an explicit `image:`
tag is set in the compose file, so the exact names depend on the folder
name Compose used as the project name when you first ran `up`.)

### Container Volumes to Delete

`docker compose down -v` (run with the matching `-f` flags above) already
removes the **named volumes** declared inside the compose file(s) -- the
three "Docker Volumes" below should already be gone once that command
succeeds. The list is kept here as a safety net for anything left behind by
an earlier run, not as a required extra step; an empty result from
`docker volume ls` afterward means cleanup worked, not that a step was
skipped.

- From the Project Repository (plain folders, delete with your OS's normal
  delete, not a `docker` command): `container-volumes\nginx\certs`
- From the Project Repository: `container-volumes\ubuntu\home-student`
- From Docker Volumes (if still present after `down -v`): `servingmlmodels_home-student`
- From Docker Volumes (if still present after `down -v`): `nginx-certs`
- From Docker Volumes (if still present after `down -v`): `nginx-frontend`

```shell
docker volume rm servingmlmodels_home-student nginx-certs nginx-frontend
```

An alternative to the manual steps outlined above is to execute the following
script:

```shell
# This is executed to create the required volume directories
chmod u+x lab_teardown.sh
sed -i 's/\r$//' lab_teardown.sh docker-compose-dev.yaml
./lab_teardown.sh
```

## Environment Variables in the `.env` File

- Delete `.env` (it holds local secrets/config and should never be committed
  to GitHub in the first place -- confirm it is listed in `.gitignore`).

## Python Virtual Environment

- Delete `.venv`.

## Hugging Face Space

Once your domain expert(s) have finished testing the multi-tab demo, either
delete the Space entirely or make it private -- there is no ongoing cost
risk on the free tier either way (ZeroGPU quota is a per-day allowance that
simply resets, not something that accumulates a bill), but a public Space
left running indefinitely is still a stale, unmaintained artifact with your
name on it.

- Go to your Space's page and click **Settings**
- Either:
  - Scroll to the **"Danger Zone"** and click **"Delete this Space"**, typing
    the Space's full name to confirm, **or**
  - Change **Visibility** from "Public" to "Private" if you would rather
    keep the Space for your own reference without it being publicly reachable

## Streamlit Community Cloud App

- Go to [https://share.streamlit.io/](https://share.streamlit.io/) and sign in
- Find your app in the list, click the **"⋮"** (three-dot) menu next to it
- Select **"Delete app"** and confirm

This removes the deployed app only -- it does not touch your GitHub
repository, so the source code being graded is unaffected.

## Render Web Service

Render's free tier costs nothing to leave running (it simply sleeps when
idle), so deleting it is about good hygiene -- removing a public API
endpoint you no longer need running under your name -- rather than avoiding
a bill.

- Go to [https://dashboard.render.com](https://dashboard.render.com)
- Select your web service
- Go to **Settings**, scroll to the bottom, and click **"Delete Web Service"**
- Confirm by typing the service's name

If you would rather keep the service available for later without deleting
it outright, you do not need to do anything extra: Render's free-tier
services already sleep automatically after 15 minutes of inactivity, so
there is no separate "pause" step required.
