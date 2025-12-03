$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item "pipetting_data/pipetting_wizard.py" "backups/pipetting_wizard_backup_$timestamp.py"