#!/bin/bash
# Deploy MORENA Pay. Syncs the UI first, because pay/ui is a copy of local/ui and a deploy that
# quietly ships a stale page is worse than one that fails.
set -euo pipefail
cd "$(dirname "$0")"
cp ../local/ui/index.html ../local/ui/console.html ui/
echo "synced $(md5sum ui/index.html | cut -c1-12)"
# Stop first, every time. A warm container keeps serving the previous build's mounted files, so a
# deploy reports success and the URL does not change. This cost three "deployed" messages that
# were all lies before it was noticed.
~/.local/bin/modal app stop morena-pay -y 2>/dev/null || true
sleep 6
~/.local/bin/modal deploy morena_pay.py
sleep 4
live=$(curl -s -m 90 "https://vambo--morena-pay-pay-web.modal.run/?cb=$RANDOM" | md5sum | cut -c1-12)
want=$(md5sum ui/index.html | cut -c1-12)
[ "$live" = "$want" ] && echo "VERIFIED live=$live" || { echo "STALE: live=$live want=$want"; exit 1; }
