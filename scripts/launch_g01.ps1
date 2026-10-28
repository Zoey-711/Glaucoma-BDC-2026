$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$smokeResult = Get-Content -LiteralPath 'outputs/phase2_g01_execution/smoke_result.json' -Raw -Encoding UTF8 | ConvertFrom-Json
if ($smokeResult.status -ne 'passed') { throw 'G01 smoke must pass first' }
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$logsPath = Join-Path $projectRoot 'outputs/phase2_g01_execution'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-u', '-m', 'scripts.g01_phase2', 'run') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logsPath "G01_$stamp.stdout.log") -RedirectStandardError (Join-Path $logsPath "G01_$stamp.stderr.log") -PassThru
@{pid=$process.Id; experiment='G01'; stdout="outputs/phase2_g01_execution/G01_$stamp.stdout.log"; stderr="outputs/phase2_g01_execution/G01_$stamp.stderr.log"} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logsPath 'process.json') -Encoding UTF8
Write-Output "G01 process launched: $($process.Id)"
