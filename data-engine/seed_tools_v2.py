#!/usr/bin/env python3
"""Build tool-calling jobs whose menus are all DIFFERENT.

docs/44 measured the failure this exists to fix: every previous toolcall example used one fixed
menu, so the model memorised that vocabulary and scored 9% on a real Paystack menu. Here each job
carries a freshly synthesised menu, with the naming convention and the argument names varied
independently, so the only way to answer is to read the schema in front of you.
"""
import json, random, sys, itertools

random.seed(20260919)
LANGS = ["pcm", "yor", "ibo", "hau", "swh", "sna", "zul", "eng"]
NG = ["pcm", "yor", "ibo", "hau"]          # Nigerian focus, weighted heavier

# concept -> (description, canonical args). The four marked GAP are the ones docs/44 found the
# model has no analogue for at all, so they get extra weight.
CONCEPTS = {
 "transfer":   ("Send money from the balance to a saved recipient", ["amount","recipient","reason","currency"], 3),
 "balance":    ("Check the available balance on the account", [], 2),
 "verify":     ("Check whether a payment went through, using its reference", ["reference"], 2),
 "refund":     ("Refund a transaction, fully or partially", ["transaction","amount","currency"], 4),      # GAP
 "resolve":    ("Look up the account holder name for a bank account number", ["account_number","bank_code"], 5),  # GAP, worst
 "recipient":  ("Save a bank account as a payout recipient", ["type","name","account_number","bank_code"], 4),     # GAP
 "vaccount":   ("Create a dedicated virtual account number for a customer", ["customer","preferred_bank"], 4),     # GAP
 "banks":      ("List the banks available in a country", ["country"], 1),
 "initpay":    ("Start a payment and get a checkout link", ["email","amount","currency"], 2),
 "customer":   ("Create a customer record", ["email","first_name","last_name"], 1),
 "airtime":    ("Buy airtime or a data bundle for a phone number", ["amount","phone","network"], 2),
 "bill":       ("Pay a utility or service bill", ["biller","account","amount"], 2),
}
# Naming conventions. The model must not be able to guess the name from the concept.
STYLES = [
 ("paystack",  {"transfer":"transfer_initiate","balance":"balance_fetch","verify":"transaction_verify","refund":"refund_create","resolve":"bank_resolveAccountNumber","recipient":"transferrecipient_create","vaccount":"dedicatedAccount_create","banks":"bank_list","initpay":"transaction_initialize","customer":"customer_create","airtime":"airtime_purchase","bill":"bill_pay"}),
 ("verbnoun",  {"transfer":"send_money","balance":"check_balance","verify":"verify_payment","refund":"issue_refund","resolve":"lookup_account_name","recipient":"save_recipient","vaccount":"create_virtual_account","banks":"list_banks","initpay":"start_payment","customer":"add_customer","airtime":"buy_airtime","bill":"pay_bill"}),
 ("camel",     {"transfer":"initiateTransfer","balance":"fetchBalance","verify":"verifyTransaction","refund":"createRefund","resolve":"resolveAccountNumber","recipient":"createRecipient","vaccount":"createVirtualAccount","banks":"listBanks","initpay":"initializePayment","customer":"createCustomer","airtime":"purchaseAirtime","bill":"payBill"}),
 ("dotted",    {"transfer":"payouts.send","balance":"wallet.balance","verify":"charges.status","refund":"charges.refund","resolve":"accounts.resolve","recipient":"payees.add","vaccount":"vaccounts.open","banks":"banks.index","initpay":"charges.create","customer":"customers.add","airtime":"airtime.topup","bill":"bills.settle"}),
 ("terse",     {"transfer":"pay","balance":"bal","verify":"chk","refund":"rfnd","resolve":"whois","recipient":"payee","vaccount":"vacct","banks":"banks","initpay":"charge","customer":"cust","airtime":"topup","bill":"bill"}),
]
ARG_ALT = {
 "amount":["amount","value","sum","amount_kobo"],
 "recipient":["recipient","beneficiary","payee","destination","recipient_code"],
 "reason":["reason","narration","note","description"],
 "currency":["currency","curr","ccy"],
 "reference":["reference","ref","txn_ref","transaction_id"],
 "transaction":["transaction","txn","charge_id"],
 "account_number":["account_number","acct_no","nuban","account"],
 "bank_code":["bank_code","bank","institution_code","sort_code"],
 "customer":["customer","customer_code","customer_id"],
 "preferred_bank":["preferred_bank","provider","bank_slug"],
 "email":["email","email_address","customer_email"],
 "country":["country","country_code","region"],
 "phone":["phone","msisdn","phone_number"],
 "network":["network","operator","telco"],
 "biller":["biller","service","merchant"],
 "account":["account","account_ref","meter_number"],
 "type":["type","recipient_type","kind"],
 "name":["name","account_name","full_name"],
 "first_name":["first_name","firstname","given_name"],
 "last_name":["last_name","lastname","surname"],
}
NUMERIC = {"amount","value","sum","amount_kobo"}

def make_menu(rng, n=6):
    style_name, style = rng.choice(STYLES)
    concepts = rng.sample(list(CONCEPTS), n)
    tools=[]
    for c in concepts:
        desc, args, _ = CONCEPTS[c]
        picked = {}
        for a in args:
            alt = rng.choice(ARG_ALT.get(a,[a]))
            picked[alt] = "number" if alt in NUMERIC else "string"
        tools.append({"name": style[c], "description": desc, "arguments": picked})
    rng.shuffle(tools)
    return style_name, concepts, tools

EDITS = ["change the amount","change the recipient","add a reason","remove the reason",
         "correct a name they got wrong","change two fields at once","cancel the whole thing",
         "confirm and go ahead"]
AMBIG = ["two saved recipients share the same first name","the bank was not named",
         "the amount could be naira or kobo","they said 'my usual one' without saying which",
         "the currency is not stated and the account supports more than one"]
MODES = ["no_tool","benign_answer","unsafe_refuse","missing_info"]

def main(n_per_type=30, out="jobs/tools_v2_pilot.json"):
    rng = random.Random(20260919)
    jobs=[]; mid=0
    def lang(): return rng.choice(NG+NG+LANGS)   # Nigerian languages double-weighted
    # toolcall_api, weighted toward the four gap concepts
    weights = list(itertools.chain.from_iterable([[c]*w for c,(_,_,w) in CONCEPTS.items()]))
    for i in range(n_per_type):
        mid+=1; style,concepts,menu = make_menu(rng)
        tgt_concept = rng.choice([c for c in weights if c in concepts]) if any(c in concepts for c in weights) else concepts[0]
        name = dict(STYLES)[style][tgt_concept]
        jobs.append({"type":"toolcall_api","lang":lang(),"id":f"t2-api-{i:05d}","phase":5,
                     "menu":menu,"target":name,"menu_id":f"{style}-{mid}","domain":"Nigerian fintech"})
    for i in range(n_per_type):
        mid+=1; style,concepts,menu = make_menu(rng)
        tname = dict(STYLES)[style]["transfer"] if "transfer" in concepts else menu[0]["name"]
        tool = next(t for t in menu if t["name"]==tname)
        cur = {"name":tname,"arguments":{k:(5000 if v=="number" else "RCP_882") for k,v in tool["arguments"].items()}}
        jobs.append({"type":"toolpatch","lang":lang(),"id":f"t2-patch-{i:05d}","phase":5,
                     "menu":menu,"current":cur,"edit_kind":rng.choice(EDITS),"menu_id":f"{style}-{mid}"})
    for i in range(n_per_type):
        mid+=1; style,concepts,menu = make_menu(rng)
        jobs.append({"type":"toolrefuse","lang":lang(),"id":f"t2-ref-{i:05d}","phase":5,
                     "menu":menu,"mode":MODES[i%len(MODES)],"menu_id":f"{style}-{mid}"})
    for i in range(n_per_type):
        mid+=1; style,concepts,menu = make_menu(rng)
        jobs.append({"type":"tooldisambig","lang":lang(),"id":f"t2-dis-{i:05d}","phase":5,
                     "menu":menu,"target":rng.choice(menu)["name"],
                     "ambiguity":rng.choice(AMBIG),"menu_id":f"{style}-{mid}"})
    json.dump(jobs, open(out,"w"), ensure_ascii=False)
    from collections import Counter
    print(f"wrote {len(jobs)} jobs -> {out}")
    print("  by type:", dict(Counter(j["type"] for j in jobs)))
    print("  by lang:", dict(Counter(j["lang"] for j in jobs)))
    print("  distinct menus:", len({j["menu_id"] for j in jobs}))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 30)
