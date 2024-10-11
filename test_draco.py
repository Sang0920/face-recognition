import requests

def checkin_realtime(email):
    """Check the API with the provided email"""
    url = "http://ecom.draerp.vn/api/method/hrms.hr.doctype.employee_checkin.employee_checkin.checkin_realtime"
    params = {'mail': email}
    auth = ('2e90b05711771df', 'cb158377a70c90b')
    
    response = requests.get(url, params=params, auth=auth)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to check in. Status code: {response.status_code}"}

# Example usage
if __name__ == "__main__":
    email = "sangdt@draco.biz"
    result = checkin_realtime(email)
    print(result)