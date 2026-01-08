#!/bin/bash
# =============================================================================
# Utility script to check job status and view logs
# =============================================================================

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  status      Show all your running/pending jobs"
    echo "  logs        List recent log files"
    echo "  tail [job]  Tail the output of a specific job"
    echo "  cancel      Cancel all your jobs"
    echo "  clean       Remove old log files"
    echo ""
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$SCRIPT_DIR/logs"

case "$1" in
    status)
        echo "Your SLURM jobs:"
        squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"
        ;;
    logs)
        echo "Recent log files in $LOGS_DIR:"
        ls -lt "$LOGS_DIR"/*.out 2>/dev/null | head -20
        ;;
    tail)
        if [ -z "$2" ]; then
            echo "Usage: $0 tail <job_id>"
            echo "Available logs:"
            ls "$LOGS_DIR"/*.out 2>/dev/null | head -10
        else
            tail -f "$LOGS_DIR"/*_$2.out
        fi
        ;;
    cancel)
        echo "Cancelling all your jobs..."
        scancel -u $USER
        echo "Done."
        ;;
    clean)
        echo "Removing log files older than 7 days..."
        find "$LOGS_DIR" -name "*.out" -mtime +7 -delete
        find "$LOGS_DIR" -name "*.err" -mtime +7 -delete
        echo "Done."
        ;;
    *)
        usage
        ;;
esac
