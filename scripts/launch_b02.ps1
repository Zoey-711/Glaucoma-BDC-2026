$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$smokeResult = Get-Content -LiteralPath 'outputs/smoke_experiments/result.json' -Raw -Encoding UTF8 | ConvertFrom-Json
if ($smokeResult.status -ne 'passed' -or -not $smokeResult.gpu_smoke) { throw 'GPU smoke test must pass first' }
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$logsPath = Join-Path $projectRoot 'outputs/experiment_manager'
New-Item -ItemType Directory -Force -Path $logsPath | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-u', '-m', 'scripts.experiments', 'run', '--config', 'configs/experiments/B02.yaml') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logsPath "B02_$stamp.stdout.log") -RedirectStandardError (Join-Path $logsPath "B02_$stamp.stderr.log") -PassThru
@{pid=$process.Id; experiment='B02'; stdout="outputs/experiment_manager/B02_$stamp.stdout.log"; stderr="outputs/experiment_manager/B02_$stamp.stderr.log"} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logsPath 'B02_process.json') -Encoding UTF8
Write-Output "B02 process launched: $($process.Id)"
