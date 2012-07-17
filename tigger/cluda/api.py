API_CUDA = 'cuda'
API_OCL = 'ocl'
APIS = [API_CUDA, API_OCL]

def supports_api(api):
    try:
        get_api(api)
    except ImportError:
        return False

    return True

def supported_apis():
    return [api for api in (API_CUDA, API_OCL) if supports_api(api)]

def get_api(api):
    if api == API_CUDA:
        import tigger.cluda.cuda
        return tigger.cluda.cuda
    elif api == API_OCL:
        import tigger.cluda.ocl
        return tigger.cluda.ocl
    else:
        raise Exception("Unrecognized API: " + str(api))
