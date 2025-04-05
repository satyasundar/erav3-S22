### This notebook is to take a base model(GEMMA 3) and GRPO fine tuning.

- **Sample output from Base mnodel**

```
Q: What is the sqrt of 101?
A : We can estimate the square root of 101. Since $10^2 = 100$, we know that $\sqrt{101}$ is slightly greater than 10.

We can try $10.05^2 = 101.0025$, so $\sqrt{101} \approx 10.05$.

Another approach is to use the fact that $\sqrt{101} = \sqrt{101}$.
We can approximate $\sqrt{101}$ using a calculator.
$\sqrt{101} \approx 10.0498756211$.

The square root of 101 is approximately 10.05.
However, if we are looking for an exact answer, we can say that $\sqrt{101}$ is not an integer.
It is known that $\sqrt{101} \approx 10.05$.

Final Answer: The final answer is $\boxed{10.05}$
```

- **Sample Output after GRPO training is done**

```
Q: What is the sqrt of 101?
A:
<start_working_out>
The square root of 101 is approximately 10.0498756.
<SOLUTION>
10.0498756<end_of_turn>
```
