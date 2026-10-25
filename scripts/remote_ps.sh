#!/usr/bin/env bash
# Run a PowerShell script (stdin) on the 40HX box via -EncodedCommand.
# The remote login shell prints proxy notes on every session, so anything that
# speaks a protocol over ssh (scp/sftp) is unusable; a base64 UTF-16LE script
# sidesteps both the banner and the quoting rules.
set -euo pipefail
enc="$(python3 -c 'import base64,sys; print(base64.b64encode(sys.stdin.read().encode("utf-16-le")).decode())')"
ssh -o BatchMode=yes "${GO_REMOTE:-ethan@192.168.1.102}" "powershell -NoProfile -EncodedCommand ${enc}"
