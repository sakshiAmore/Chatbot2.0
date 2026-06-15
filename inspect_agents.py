import langchain.agents as agents
print('file=', agents.__file__)
print('initialize_agent' in dir(agents))
print([x for x in dir(agents) if 'initialize' in x.lower() or 'agent' in x.lower()][:100])
