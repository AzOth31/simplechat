# lambda/index.py
import json
import os
import re  # 正規表現モジュールをインポート
import urllib.request #追加
import urllib.error
import time

# Ngrok URL は環境変数で上書き可能に
NGROK_URL = os.environ.get(
    "NGROK_URL",
    "https://9772-104-196-41-144.ngrok-free.app"
)

class LLMClient:
    """LLM API クライアントクラス"""
    
    def __init__(self, api_url):
        """
        初期化
        Args:
            api_url (str): API のベース URL（ngrok URL）
        """
        self.api_url = api_url.rstrip('/')
        #self.session = requests.Session()
    
    def generate(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True) -> dict:
        """
        テキスト生成
        Args:
            prompt (str): プロンプト文字列
            max_new_tokens (int, optional): 生成する最大トークン数
            temperature (float, optional): 温度パラメータ
            top_p (float, optional): top-p サンプリングのパラメータ
            do_sample (bool, optional): サンプリングを行うかどうか
        Returns:
            dict: 生成結果
        """
        url = f'{self.api_url}/generate'
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                result = json.loads(body)
                result["total_request_time"] = time.time() - start
                return result
        except urllib.error.HTTPError as e:
            raise Exception(f"Generate HTTP error: {e.code} - {e.reason}")
        except urllib.error.URLError as e:
            raise Exception(f"Generate network error: {e.reason}")

client = LLMClient(NGROK_URL)

# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    # ARN 形式: arn:aws:lambda:region:account-id:function:function-name
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値

def lambda_handler(event, context):
    try:
        # コンテキストから実行リージョンを取得し、クライアントを初期化
        global client
        if client is None:
            region = extract_region_from_arn(context.invoked_function_arn)
            #bedrock_client = boto3.client('bedrock-runtime', region_name=region)
            print(f"Initialized Bedrock client in region: {region}")
        
        print("Received event:", json.dumps(event))
        
        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        
        print("Processing message:", message)

        # 会話履歴を使用
        messages = conversation_history.copy()
        
        # ユーザーメッセージを追加
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Nova Liteモデル用のリクエストペイロードを構築
        # 会話履歴を含める
        response = client.generate(message)
        assistant_response = response.get("generated_text") or response.get("text") or ""
        
        # アシスタントの応答を会話履歴に追加
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }
        
    except Exception as error:
        print("Error:", str(error))
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }
