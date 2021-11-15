def create_payment_transaction():
    import random
    import string
    
    valid_type = ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']
    account_string = ['C', 'M']

    v_type = random.choice(valid_type)
    string_start = random.choice(account_string)

    q = ''.join(random.choices(string.digits, k=11))
    q2 = ''.join(random.choices(string.digits, k=11))
    string_start2 = random.choice(account_string)

    account_name = string_start.__add__(q)

    step_ = 0

    amount_ = random.randint(0,1000000000)

    oldbalanceOrg_ = random.randint(0,1000000000)
    newbalanceOrig_ = abs(oldbalanceOrg_ - amount_)

    account_dest = string_start2.__add__(q2)

    oldbalancedest_ = random.randint(0,1000000000)
    newbalancedest_ = oldbalancedest_ + amount_

    isfraud_ = random.randint(0,1)

    if (amount_ > 200000) == True:
        isflagged = 1
    else:
        isflagged = 0


    return {'step': step_, 'type': v_type, 'amount': amount_, 'nameOrig': account_name, 'oldbalanceOrg': oldbalanceOrg_, 
        'newbalanceOrig': newbalanceOrig_, 'nameDest': account_dest, 'oldbalanceDest': oldbalancedest_, 'newbalanceDest': newbalancedest_, 'isFraud': isfraud_, 'isFlaggedFraud': isflagged}
