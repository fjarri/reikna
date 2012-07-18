API_CUDA = 'cuda'
API_OCL = 'ocl'
APIS = [API_CUDA, API_OCL]

def supports_api(api_id):
    try:
        get_api(api_id)
    except ImportError:
        return False

    return True

def supported_apis():
    return [api_id for api_id in (API_CUDA, API_OCL) if supports_api(api_id)]

def api(api_id):
    if api_id == API_CUDA:
        import tigger.cluda.cuda
        return tigger.cluda.cuda
    elif api_id == API_OCL:
        import tigger.cluda.ocl
        return tigger.cluda.ocl
    else:
        raise Exception("Unrecognized API: " + str(api_id))
