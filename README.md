# interactive_yuma_simulator

Interactive Yuma Consensus Simulator

- - -

# Base requirements

- docker with [compose plugin](https://docs.docker.com/compose/install/linux/)
- python 3.11
- [uv](https://docs.astral.sh/uv/)
- [nox](https://nox.thea.codes)

# Setup development environment

```sh
./setup-dev.sh
docker compose up -d
cd app/src
uv run manage.py wait_for_database --timeout 10
uv run manage.py migrate
uv run manage.py runserver
```

# Setup production environment (git deployment)

<details>

This sets up "deployment by pushing to git storage on remote", so that:

- `git push origin ...` just pushes code to Github / other storage without any consequences;
- `git push production master` pushes code to a remote server running the app and triggers a git hook to redeploy the application.

```
Local .git ------------> Origin .git
                \
                 ------> Production .git (redeploy on push)
```

- - -

Use `ssh-keygen` to generate a key pair for the server, then add read-only access to repository in "deployment keys" section (`ssh -A` is easy to use, but not safe).

```sh
# remote server
mkdir -p ~/repos
cd ~/repos
git init --bare --initial-branch=master interactive_yuma_simulator.git

mkdir -p ~/domains/interactive_yuma_simulator
```

```sh
# locally
git remote add production root@<server>:~/repos/interactive_yuma_simulator.git
git push production master
```

```sh
# remote server
cd ~/repos/interactive_yuma_simulator.git

cat <<'EOT' > hooks/post-receive
#!/bin/bash
unset GIT_INDEX_FILE
export ROOT=/root
export REPO=interactive_yuma_simulator
while read oldrev newrev ref
do
    if [[ $ref =~ .*/master$ ]]; then
        export GIT_DIR="$ROOT/repos/$REPO.git/"
        export GIT_WORK_TREE="$ROOT/domains/$REPO/"
        git checkout -f master
        cd $GIT_WORK_TREE
        ./deploy.sh
    else
        echo "Doing nothing: only the master branch may be deployed on this server."
    fi
done
EOT

chmod +x hooks/post-receive
./hooks/post-receive
cd ~/domains/interactive_yuma_simulator
sudo bin/prepare-os.sh
./setup-prod.sh

# adjust the `.env` file

mkdir letsencrypt
./letsencrypt_setup.sh
./deploy.sh
```

### Deploy another branch

Only `master` branch is used to redeploy an application.
If one wants to deploy other branch, force may be used to push desired branch to remote's `master`:

```sh
git push --force production local-branch-to-deploy:master
```

</details>



# Cloud deployment

## AWS

<details>
Initiate the infrastructure with Terraform:
TODO

To push a new version of the application to AWS, just push to a branch named `deploy-$(ENVIRONMENT_NAME)`.
Typical values for `$(ENVIRONMENT_NAME)` are `prod` and `staging`.
For this to work, GitHub actions needs to be provided with credentials for an account that has the following policies enabled:

- AutoScalingFullAccess
- AmazonEC2ContainerRegistryFullAccess
- AmazonS3FullAccess

See `.github/workflows/cd.yml` to find out the secret names.

For more details see [README_AWS.md](README_AWS.md)
</details>

## Vultr

<details>
Initiate the infrastructure with Terraform and cloud-init:

- see Terraform template in `<project>/devops/vultr_tf/core/`
- see scripts for interacting with Vultr API in `<project>/devops/vultr_scripts/`
  - note these scripts need `vultr-cli` installed

For more details see [README_vultr.md](README_vultr.md).
</details>

# Backups

<details>
<summary>Click to for backup setup & recovery information</summary>

Backups are managed by `backups` container.

## Local volume

By default, backups will be created [periodically](backups/backup.cron) and stored in `backups` volume.

### Backups rotation
Set env var:
- `BACKUP_LOCAL_ROTATE_KEEP_LAST`

### Email

Local backups may be sent to email manually. Set env vars:
- `EMAIL_HOST`
- `EMAIL_PORT`
- `EMAIL_HOST_USER`
- `EMAIL_HOST_PASSWORD`
- `DEFAULT_FROM_EMAIL`

Then run:
```sh
docker compose run --rm -e EMAIL_TARGET=youremail@domain.com backups ./backup-db.sh
```

## B2 cloud storage

Create an application key with restricted access to a single bucket. The key should have following permissions:
- `listFiles`
- `readFiles`
- `writeFiles`

Set env vars:
- `BACKUP_B2_BUCKET`
- `BACKUP_B2_KEY_ID`
- `BACKUP_B2_KEY_SECRET`

Backups will be uploaded to the bucket.

Set up backups lifecycle rules by following these instructions: https://www.backblaze.com/docs/cloud-storage-configure-and-manage-lifecycle-rules

## List all available backups

```sh
docker compose run --rm backups ./list-backups.sh
```

## Restoring system from backup after a catastrophical failure

1. Follow the instructions above to set up a new production environment
2. Restore the database using one of
```sh
docker compose run --rm backups ./restore-db.sh /var/backups/{backup-name}.dump.zstd

docker compose run --rm backups ./restore-db.sh b2://{bucket-name}/{backup-name}.dump.zstd
docker compose run --rm backups ./restore-db.sh b2id://{ID}
```
3. See if everything works
4. Make sure everything is filled up in `.env`, error reporting integration, email accounts etc

## Monitoring

`backups` container runs a simple server which [exposes essential metrics about backups](backups/bin/serve_metrics.py).

</details>

# cookiecutter-rt-django

Skeleton of this project was generated using [cookiecutter-rt-django](https://github.com/reef-technologies/cookiecutter-rt-django).
Use `cruft update` to update the project to the latest version of the template with all current bugfixes and [features](https://github.com/reef-technologies/cookiecutter-rt-django/blob/master/features.md).
