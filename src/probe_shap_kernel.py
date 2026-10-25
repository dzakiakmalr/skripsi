"""Read-only inspection of an existing kernel; never fit or replay cells."""
import json
import sys
from jupyter_client import BlockingKernelClient

client = BlockingKernelClient(connection_file=sys.argv[1])
client.load_connection_file()
client.start_channels()
try:
    client.wait_for_ready(timeout=10)
    msg_id = client.execute('', silent=True, store_history=False, user_expressions={
        'state': "{k: (type(globals()[k]).__name__, len(globals()[k]) if hasattr(globals()[k], '__len__') else None) for k in ['searches','results','predictions','probabilities','X_test','y_test','FEATURES','RUN_DIR','assert_original_data'] if k in globals()}",
        'models': "list(searches.keys()) if 'searches' in globals() else []",
        'results': "results[['Model','Imbalance Method','F1','Recall','ROC-AUC']].to_dict('records') if 'results' in globals() else []",
    })
    while True:
        reply = client.get_shell_msg(timeout=15)
        if reply['parent_header'].get('msg_id') == msg_id:
            print(json.dumps(reply['content'], ensure_ascii=True, indent=2))
            break
finally:
    client.stop_channels()
