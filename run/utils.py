# def run_explicit_model_instance():
#     # ------------------- SYMBOLIC SETUP -------------------
#     x1, x2, b = symbols('x1 x2 b')
#     x_syms = [x1, x2]
#     residuals = Matrix([x1 ** 2 + x2 - b, x1 + x2 ** 2 - b])

#     # Solve for x1, x2 in terms of b
#     solutions = solve(residuals, x_syms, dict=True)

#     # Pick only real solutions with simple form (e.g., avoid complex branches)
#     sol_fn = lambdify(b, [solutions[0][x1], solutions[0][x2]], modules='numpy')

#     # Dataset Creation and Model Running
#     dataset = BDataset(sol_fn=sol_fn)
#     train_len = int(0.8 * len(dataset))
#     test_len = len(dataset) - train_len
#     train_set, test_set = random_split(dataset, [train_len, test_len])

#     train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
#     val_loader = DataLoader(test_set, batch_size=64)
#     test_loader = DataLoader(test_set, batch_size=64)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = NewtonModel(residuals, x_syms, [b]).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.MSELoss()

#     trainer = KKT_HardNet_Trainer(model, train_loader, val_loader, test_loader, optimizer, criterion,
#                                   num_epochs=700, eta=1e-3, model_loss_tolerance=1e-6)
#     trainer.train()
import re

def categorize(sym):
    name = str(sym)

    # Pure y variables like y1, y2
    if re.fullmatch(r"y\d+", name):
        return (0, 0, name)
    
    elif name.startswith("y") and not (name.endswith("data") or name.endswith("delta")):
        return (1, 0, name)
    
    elif name.startswith("d") and not name.endswith("data"):
        return (2, 0, name)

    # # Other y-prefixed variables
    # elif name.startswith("y") and not name.endswith("data"):
    #     return (3, 0, name)

    elif name.startswith("mu"):
        return (3, 0, name)
    elif name.startswith("s"):
        return (4, 0, name)
    elif name.startswith("delta"):
        return (5, 0, name)
    elif name.startswith("sigma"):
        return (6, 0, name)
    elif name.startswith("lambda"):
        return (7, 0, name)
    elif name.startswith("t") and not name.endswith("delta"):
        return (8, 0, name)
    elif name.startswith("x") and not name.endswith("delta"):
        return (9, 0, name)
    elif name.startswith("y") and name.endswith("data"):
        return (10, 0, name)
    elif name.startswith("d") and name.endswith("data"):
        return (11, 0, name)
    elif name.endswith("delta"):
        return (12, 0, name)
    else:
        return (13, 0, name)