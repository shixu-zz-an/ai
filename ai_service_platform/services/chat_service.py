import json
import requests
from config import QWEN_API_URL, QWEN_MODEL

class ChatService:
    def __init__(self):
        self.history = []
        
    def chat(self, message: str):
        """与 Qwen 进行对话"""
        try:
            # 构建请求数据
            request_data = {
                "model": QWEN_MODEL,
                "messages": self.history + [{"role": "user", "content": message}],
                "stream": True
            }
            
            # 发送请求到 Qwen API
            response = requests.post(
                QWEN_API_URL,
                json=request_data,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
      
            # 用于累积完整的回复
            full_response = ""             
            # 处理流式响应
            for line in response.iter_lines():
                if not line:
                    continue
                    
                try:
                    # 解码二进制数据
                    line_str = line.decode('utf-8')
                    if line_str.startswith('b\''):
                        # 去除b''
                        line_str = line_str[2:-1]
                        # 处理转义字符
                        line_str = bytes(line_str, 'utf-8').decode('unicode_escape')
                    
                    # 解析JSON
                    data = json.loads(line_str)
                    print(f"收到数据行: {line}")  # 调试日志
                    
                    if data.get('done', False):
                        # 更新对话历史
                        self.history.append({"role": "user", "content": message})
                        if full_response:
                            self.history.append({"role": "assistant", "content": full_response})
                        
                        # 保持历史记录在合理范围内
                        if len(self.history) > 10:
                            self.history = self.history[-10:]
                        
                        yield 'data: [DONE]\n\n'
                        break
                        
                    if 'message' in data:
                        content = data['message'].get('content', '')
                        if content:
                            full_response += content
                            # 发送内容给前端
                            yield f'data: {json.dumps({"answer": content}, ensure_ascii=False)}\n\n'
                            
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {str(e)}, 原始数据: {line}")
                    continue
                except Exception as e:
                    print(f"处理响应行时出错: {str(e)}, 原始数据: {line}")
                    yield f'data: {json.dumps({"error": str(e)}, ensure_ascii=False)}\n\n'
                    continue
                
        except Exception as e:
            error_msg = f"对话处理失败: {str(e)}"
            print(f"错误: {error_msg}")
            yield f'data: {json.dumps({"error": error_msg}, ensure_ascii=False)}\n\n'
            yield 'data: [DONE]\n\n'
    
    def clear_history(self):
        """清空对话历史"""
        self.history = []
