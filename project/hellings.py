from pyformlang.cfg import CFG, Epsilon, Production, Variable

def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    cnf = cfg.to_normal_form()
    prods = set(cnf.prods)

    for var in cfg.get_nullable_symbols():
        v = Variable(var.value)
        prods.add(Production(v, []))
        prods.add(Production(v, [Epsilon()]))

    wcnf = CFG(start_symbol=cfg.start_symbol, productions=prods)

    return wcnf