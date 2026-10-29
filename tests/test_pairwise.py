import pytest
import torch
from torch.nn import functional as F

from src.losses import BinaryLoss
from src.pairwise import ranking_term, bce_pairwise, LossAccounting, monitor_training


def test_all_pairs_match_explicit_enumeration_and_gradient_direction():
    z = torch.tensor([-.5, 2., 1., -2.], requires_grad=True)
    y = torch.tensor([1., 1., 0., 0.])
    rank, count = ranking_term(z, y)
    explicit = torch.stack([F.softplus(z[j]-z[i]) for i in [0,1] for j in [2,3]]).mean()
    assert count == 4 and torch.equal(rank, explicit)
    rank.backward()
    assert (z.grad[:2] < 0).all() and (z.grad[2:] > 0).all()


def test_pairwise_gradcheck_and_extreme_logits():
    z = torch.tensor([.3, -.8, 1.4], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([1., 0., 0.], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: bce_pairwise(x,y,.05), (z,))
    extreme = torch.tensor([-10000.,10000.],requires_grad=True)
    loss = bce_pairwise(extreme,torch.tensor([1.,0.]),.2)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(extreme.grad).all()


@pytest.mark.parametrize('label',[0.,1.])
def test_single_class_exact_bce_and_accounting(label):
    z=torch.tensor([2.,-1.],requires_grad=True);y=torch.full((2,),label)
    monitor=LossAccounting()
    with monitor_training(monitor):
        loss=bce_pairwise(z,y,.2)
    assert torch.equal(loss,F.binary_cross_entropy_with_logits(z,y))
    loss.backward(); assert torch.isfinite(z.grad).all()
    row=monitor.finish_epoch()
    assert row['valid_batch_fraction']==0 and row['effective_pair_count']==0 and row['weighted_pairwise_loss']==0


def test_lambda_zero_matches_bce_values_gradients_and_optimizer_updates():
    torch.manual_seed(42)
    a=torch.nn.Linear(4,1);b=torch.nn.Linear(4,1);b.load_state_dict(a.state_dict())
    oa=torch.optim.AdamW(a.parameters(),lr=.0003,weight_decay=.0001)
    ob=torch.optim.AdamW(b.parameters(),lr=.0003,weight_decay=.0001)
    loss_a=BinaryLoss({'name':'bce'},[6,2]);loss_b=BinaryLoss({'name':'pairwise_auc','lambda':0},[6,2])
    for _ in range(3):
        x=torch.randn(8,4);y=torch.tensor([0.,0.,0.,1.,0.,0.,1.,0.])
        oa.zero_grad();ob.zero_grad()
        la=loss_a(a(x).flatten(),y);lb=loss_b(b(x).flatten(),y)
        assert torch.equal(la,lb)
        la.backward();lb.backward()
        assert all(torch.equal(pa.grad,pb.grad) for pa,pb in zip(a.parameters(),b.parameters()))
        oa.step();ob.step()
        assert all(torch.equal(pa,pb) for pa,pb in zip(a.parameters(),b.parameters()))


def test_epoch_loss_decomposition_matches_samples():
    monitor=LossAccounting()
    with monitor_training(monitor):
        bce_pairwise(torch.tensor([1.,-1.,0.]),torch.tensor([1.,0.,0.]),.05)
        bce_pairwise(torch.tensor([2.,3.]),torch.tensor([0.,0.]),.05)
    r=monitor.finish_epoch()
    assert r['positive_draws']==1 and r['negative_draws']==4 and r['effective_pair_count']==2
    assert r['valid_batch_fraction']==.5
    assert abs(r['combined_loss']-r['bce_loss']-r['weighted_pairwise_loss'])<1e-12
