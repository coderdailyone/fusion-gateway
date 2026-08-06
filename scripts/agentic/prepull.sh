#!/usr/bin/env bash
export XDG_RUNTIME_DIR="/run/user/$(id -u)"; export DOCKER_HOST="unix://${XDG_RUNTIME_DIR}/docker.sock"; export PATH="$HOME/bin:$PATH"
imgs=$(~/m4/.venv/bin/python -c "import json;[print(r['image_name']) for r in json.load(open('/home/gaozhi/m4/inst_pilot.json'))]")
echo "start $(date +%T) pulling $(echo "$imgs"|wc -l) images"
echo "$imgs" | xargs -P 4 -I {} sh -c 'docker pull {} >/dev/null 2>&1 && echo "ok {}" || echo "FAIL {}"'
echo "end $(date +%T)"
have=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -c starryzhang)
echo "PREPULL_DONE have=$have starryzhang images"
