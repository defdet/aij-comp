def format_replic(replic):
    prompt = []
    for data in replic:
        data_type, content = data['type'], data['content']
        if data_type == 'text':
            prompt.append(f"User: {content}")
        elif data_type == 'image':
            prompt.append(content)
        elif data_type == 'audio':
            captions = get_audio(content)
            prompt.append("Since you cannot listen to audio, here's audio captioning: {captions}"
    prompt.append("<end_of_utterance>")
    prompt.append('\nAssistant:')
