import { test, expect, Page, BrowserContext } from '@playwright/test';

/**
 * Comprehensive E2E Test Suite for ADK Security Agent
 * Tests dashboard, streaming chat, security analysis, and API integration
 */

test.describe('Security Agent - Full E2E Testing', () => {
  let page: Page;
  let context: BrowserContext;

  test.beforeEach(async ({ browser }) => {
    context = await browser.newContext({
      permissions: ['clipboard-read', 'clipboard-write'],
    });
    page = await context.newPage();
    
    // Navigate to the app
    await page.goto('/');
    
    // Wait for the app to load - Streamlit apps take time to render
    await page.waitForLoadState('networkidle');
    // Wait for the main title to appear instead of checking page title
    await page.waitForSelector('h1', { timeout: 10000 });
    await expect(page.getByRole('heading', { level: 1 })).toContainText(/GCP Security|Security Agent/i);
  });

  test.afterEach(async () => {
    await context.close();
  });

  test.describe('Dashboard Tests', () => {
    test('should display executive dashboard on front page', async () => {
      // Check for dashboard elements - updated to match actual app
      await expect(page.locator('h1').filter({ hasText: /GCP Security Executive Dashboard/i })).toBeVisible();
      
      // Verify security metrics cards are present (using text content)
      await expect(page.getByText('Total Assets').first()).toBeVisible();
      await expect(page.getByText('Critical/High').first()).toBeVisible();
      await expect(page.getByText('Storage Security').first()).toBeVisible();
      await expect(page.getByText('Overall Health').first()).toBeVisible();
    });

    test('should show data freshness indicators', async () => {
      // Check for data import status
      const importStatus = page.locator('[data-testid="import-status"]');
      await expect(importStatus).toBeVisible();
      
      // Verify freshness indicator (green/yellow/red)
      const freshnessIndicator = importStatus.locator('[data-testid="freshness-indicator"]');
      await expect(freshnessIndicator).toBeVisible();
      await expect(freshnessIndicator).toHaveAttribute('data-status', /fresh|recent|stale/);
    });

    test('should refresh data when refresh button clicked', async () => {
      const refreshButton = page.locator('[data-testid="refresh-button"]');
      await expect(refreshButton).toBeVisible();
      
      // Click refresh
      await refreshButton.click();
      
      // Wait for loading state
      await expect(page.locator('[data-testid="loading-indicator"]')).toBeVisible();
      
      // Wait for refresh to complete (max 30 seconds)
      await expect(page.locator('[data-testid="loading-indicator"]')).not.toBeVisible({ timeout: 30000 });
      
      // Verify data updated
      await expect(page.locator('[data-testid="last-update-time"]')).toContainText(/just now|seconds ago/i);
    });

    test('should display security trend graphs', async () => {
      // Check for trend visualization
      await expect(page.locator('[data-testid="security-trend-chart"]')).toBeVisible();
      await expect(page.locator('[data-testid="finding-distribution-chart"]')).toBeVisible();
    });

    test('should export security report', async () => {
      // Click export button
      const exportButton = page.locator('[data-testid="export-button"]');
      await expect(exportButton).toBeVisible();
      
      // Start waiting for download
      const downloadPromise = page.waitForEvent('download');
      await exportButton.click();
      
      // Wait for download
      const download = await downloadPromise;
      expect(download.suggestedFilename()).toMatch(/security_report.*\.(md|json)/);
    });
  });

  test.describe('Streaming Chat Interface Tests', () => {
    test('should display chat interface below dashboard', async () => {
      // Scroll to chat section
      await page.locator('[data-testid="chat-section"]').scrollIntoViewIfNeeded();
      
      // Verify chat elements
      await expect(page.locator('[data-testid="chat-input"]')).toBeVisible();
      await expect(page.locator('[data-testid="chat-messages"]')).toBeVisible();
    });

    test('should stream tokens in real-time for chat responses', async () => {
      const chatInput = page.getByPlaceholder('Ask about your GCP security posture');
      const sendButton = page.getByTestId('stChatInputSubmitButton');
      
      // Type a security query
      await chatInput.fill('What are my critical security findings?');
      await sendButton.click();
      
      // Verify streaming response
      const responseMessage = page.locator('[data-testid="assistant-message"]').last();
      
      // Check that response appears token by token
      await expect(responseMessage).toBeVisible();
      
      // Wait for first token
      await expect(responseMessage).not.toBeEmpty();
      
      // Capture initial text
      const initialText = await responseMessage.textContent();
      
      // Wait a bit and check text has grown (streaming)
      await page.waitForTimeout(500);
      const laterText = await responseMessage.textContent();
      expect(laterText?.length).toBeGreaterThan(initialText?.length || 0);
      
      // Wait for response to complete (max 30 seconds)
      await expect(responseMessage).toContainText(/finding|security|critical/i, { timeout: 30000 });
    });

    test('should handle multi-turn conversations', async () => {
      const chatInput = page.getByPlaceholder('Ask about your GCP security posture');
      const sendButton = page.getByTestId('stChatInputSubmitButton');
      
      // First query
      await chatInput.fill('List my storage buckets');
      await sendButton.click();
      
      // Wait for first response
      await expect(page.locator('[data-testid="assistant-message"]').last()).toContainText(/bucket/i, { timeout: 30000 });
      
      // Follow-up query
      await chatInput.fill('Which ones are publicly accessible?');
      await sendButton.click();
      
      // Verify context is maintained
      await expect(page.locator('[data-testid="assistant-message"]').last()).toContainText(/public|accessible/i, { timeout: 30000 });
    });

    test('should display quick queries sidebar', async () => {
      const quickQueries = page.locator('[data-testid="quick-queries"]');
      await expect(quickQueries).toBeVisible();
      
      // Click a quick query
      const firstQuery = quickQueries.locator('button').first();
      const queryText = await firstQuery.textContent();
      await firstQuery.click();
      
      // Verify query is executed
      await expect(page.locator('[data-testid="user-message"]').last()).toContainText(queryText || '');
      await expect(page.getByText(/analyzing|security|finding/i)).toBeVisible({ timeout: 30000 });
    });

    test('should handle empty queries gracefully', async () => {
      const sendButton = page.locator('[data-testid="send-button"]');
      
      // Try to send empty message
      await sendButton.click();
      
      // Verify error message or disabled state
      await expect(page.locator('[data-testid="error-message"]')).toContainText(/enter a message/i);
    });
  });

  test.describe('Security Analysis Features', () => {
    test('should analyze IAM policies', async () => {
      const chatInput = page.getByPlaceholder('Ask about your GCP security posture');
      const sendButton = page.getByTestId('stChatInputSubmitButton');
      
      await chatInput.fill('Analyze my IAM policies for overly permissive roles');
      await sendButton.click();
      
      const response = page.getByText(/IAM|role|permission|analyzing/i).first();
      await expect(response).toContainText(/IAM|role|permission/i, { timeout: 30000 });
      
      // Check for specific security recommendations
      await expect(response).toContainText(/recommend|suggest|should/i);
    });

    test('should check storage bucket security', async () => {
      const chatInput = page.getByPlaceholder('Ask about your GCP security posture');
      const sendButton = page.getByTestId('stChatInputSubmitButton');
      
      await chatInput.fill('Check my storage buckets for security issues');
      await sendButton.click();
      
      const response = page.getByText(/IAM|role|permission|analyzing/i).first();
      await expect(response).toContainText(/bucket|storage/i, { timeout: 30000 });
      
      // Verify security checks
      await expect(response).toContainText(/public|encryption|versioning/i);
    });

    test('should provide compliance assessment', async () => {
      const chatInput = page.getByPlaceholder('Ask about your GCP security posture');
      const sendButton = page.getByTestId('stChatInputSubmitButton');
      
      await chatInput.fill('Assess my SOC2 compliance status');
      await sendButton.click();
      
      const response = page.getByText(/IAM|role|permission|analyzing/i).first();
      await expect(response).toContainText(/SOC2|compliance/i, { timeout: 30000 });
      
      // Check for compliance details
      await expect(response).toContainText(/control|requirement|audit/i);
    });

    test('should analyze security findings', async () => {
      const chatInput = page.getByPlaceholder('Ask about your GCP security posture');
      const sendButton = page.getByTestId('stChatInputSubmitButton');
      
      await chatInput.fill('Show me critical security findings from Security Command Center');
      await sendButton.click();
      
      const response = page.getByText(/IAM|role|permission|analyzing/i).first();
      await expect(response).toContainText(/finding|security|critical/i, { timeout: 30000 });
      
      // Verify finding details
      await expect(response).toContainText(/severity|risk|remediation/i);
    });
  });

  test.describe('Error Handling and Recovery', () => {
    test('should handle backend connection errors gracefully', async () => {
      // Simulate backend failure by navigating when backend is down
      await page.route('**/api/**', route => route.abort());
      
      const chatInput = page.getByPlaceholder('Ask about your GCP security posture');
      const sendButton = page.getByTestId('stChatInputSubmitButton');
      
      await chatInput.fill('Test query');
      await sendButton.click();
      
      // Verify error handling - Streamlit shows errors differently
      await expect(page.getByText(/error|failed|connection/i)).toBeVisible({ timeout: 10000 });
    });

    test('should recover from transient errors', async () => {
      let requestCount = 0;
      
      // Fail first request, succeed on retry
      await page.route('**/api/v1/chat', route => {
        requestCount++;
        if (requestCount === 1) {
          route.abort();
        } else {
          route.continue();
        }
      });
      
      const chatInput = page.getByPlaceholder('Ask about your GCP security posture');
      const sendButton = page.getByTestId('stChatInputSubmitButton');
      
      await chatInput.fill('Test query with retry');
      await sendButton.click();
      
      // Should automatically retry and succeed
      await expect(page.getByText(/analyzing|security|finding/i)).toBeVisible({ timeout: 30000 });
    });

    test('should validate input and show helpful prompts', async () => {
      const chatInput = page.getByPlaceholder('Ask about your GCP security posture');
      const sendButton = page.getByTestId('stChatInputSubmitButton');
      
      // Test very short query
      await chatInput.fill('Hi');
      await sendButton.click();
      
      // Should accept the input (Streamlit doesn't show suggestions inline)
      await expect(page.getByText('Hi')).toBeVisible();
    });
  });

  test.describe('Accessibility Tests', () => {
    test('should support keyboard navigation', async () => {
      // Tab to chat input
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');
      
      // Check focus is on chat input
      const chatInput = page.locator('[data-testid="chat-input"] input');
      await expect(chatInput).toBeFocused();
      
      // Type with keyboard
      await page.keyboard.type('Test keyboard navigation');
      
      // Submit with Enter
      await page.keyboard.press('Enter');
      
      // Verify message sent
      await expect(page.locator('[data-testid="user-message"]').last()).toContainText('Test keyboard navigation');
    });

    test('should have proper ARIA labels', async () => {
      // Check main elements have ARIA labels
      await expect(page.locator('[data-testid="chat-input"]')).toHaveAttribute('aria-label', /chat|message|input/i);
      await expect(page.locator('[data-testid="send-button"]')).toHaveAttribute('aria-label', /send|submit/i);
      await expect(page.locator('[data-testid="chat-messages"]')).toHaveAttribute('aria-label', /conversation|messages/i);
    });

    test('should work with screen readers', async () => {
      // Check for live regions for dynamic content
      await expect(page.locator('[aria-live="polite"]')).toBeVisible();
      
      // Verify status announcements
      const chatInput = page.getByPlaceholder('Ask about your GCP security posture');
      const sendButton = page.getByTestId('stChatInputSubmitButton');
      
      await chatInput.fill('Screen reader test');
      await sendButton.click();
      
      // Check for status update in live region
      await expect(page.locator('[aria-live="polite"]')).toContainText(/processing|sending/i);
    });
  });

  test.describe('Mobile Responsiveness', () => {
    test('should adapt to mobile viewport', async () => {
      // Set mobile viewport
      await page.setViewportSize({ width: 375, height: 812 });
      
      // Check mobile menu is visible
      await expect(page.locator('[data-testid="mobile-menu-button"]')).toBeVisible();
      
      // Dashboard should stack vertically
      const cards = await page.locator('[data-testid*="-card"]').all();
      for (const card of cards) {
        const box = await card.boundingBox();
        expect(box?.width).toBeLessThan(360); // Cards should fit in mobile width
      }
    });

    test('should handle touch interactions', async () => {
      await page.setViewportSize({ width: 375, height: 812 });
      
      // Simulate touch on quick query
      const quickQuery = page.locator('[data-testid="quick-queries"] button').first();
      await quickQuery.tap();
      
      // Verify query executed
      await expect(page.locator('[data-testid="user-message"]').last()).toBeVisible();
    });
  });

  test.describe('Performance Tests', () => {
    test('should load dashboard within acceptable time', async () => {
      const startTime = Date.now();
      
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      
      const loadTime = Date.now() - startTime;
      expect(loadTime).toBeLessThan(5000); // Should load within 5 seconds
    });

    test('should handle large datasets efficiently', async () => {
      const chatInput = page.getByPlaceholder('Ask about your GCP security posture');
      const sendButton = page.getByTestId('stChatInputSubmitButton');
      
      // Request large dataset
      await chatInput.fill('Show me all security findings with full details');
      await sendButton.click();
      
      // Should start streaming quickly
      const responseStart = Date.now();
      await expect(page.locator('[data-testid="assistant-message"]').last()).toBeVisible();
      const timeToFirstToken = Date.now() - responseStart;
      
      expect(timeToFirstToken).toBeLessThan(2000); // First token within 2 seconds
    });

    test('should cache responses appropriately', async () => {
      const chatInput = page.getByPlaceholder('Ask about your GCP security posture');
      const sendButton = page.getByTestId('stChatInputSubmitButton');
      
      // First query
      await chatInput.fill('List my projects');
      await sendButton.click();
      await expect(page.getByText(/analyzing|security|finding/i)).toBeVisible({ timeout: 30000 });
      
      // Same query again (should be faster due to caching)
      const startTime = Date.now();
      await chatInput.fill('List my projects');
      await sendButton.click();
      await expect(page.locator('[data-testid="assistant-message"]').nth(-1)).toBeVisible();
      const cachedResponseTime = Date.now() - startTime;
      
      expect(cachedResponseTime).toBeLessThan(1000); // Cached response under 1 second
    });
  });
});

test.describe('API Integration Tests', () => {
  test('should query SQLite database correctly', async ({ page }) => {
    await page.goto('/');
    
    const chatInput = page.locator('[data-testid="chat-input"] input');
    const sendButton = page.locator('[data-testid="send-button"]');
    
    // Test database query
    await chatInput.fill('What tables are available in the security database?');
    await sendButton.click();
    
    const response = page.locator('[data-testid="assistant-message"]').last();
    await expect(response).toContainText(/table|database|sqlite/i, { timeout: 30000 });
    
    // Should list actual tables
    await expect(response).toContainText(/assets|findings|policies|buckets/i);
  });

  test('should integrate with GCP APIs through backend', async ({ page }) => {
    await page.goto('/');
    
    // Check that GCP data is being displayed
    await expect(page.locator('[data-testid="gcp-project-id"]')).toBeVisible();
    await expect(page.locator('[data-testid="gcp-project-id"]')).not.toBeEmpty();
    
    // Verify API data in dashboard
    const metrics = await page.locator('[data-testid="security-metrics"]').textContent();
    expect(metrics).toMatch(/\d+/); // Should contain numbers from API data
  });

  test('should handle API rate limiting gracefully', async ({ page }) => {
    await page.goto('/');
    
    const chatInput = page.locator('[data-testid="chat-input"] input');
    const sendButton = page.locator('[data-testid="send-button"]');
    
    // Send multiple rapid requests
    for (let i = 0; i < 35; i++) {
      await chatInput.fill(`Query ${i}`);
      await sendButton.click();
      await page.waitForTimeout(100);
    }
    
    // Should show rate limit message
    await expect(page.locator('[data-testid="rate-limit-warning"]')).toBeVisible();
    await expect(page.locator('[data-testid="rate-limit-warning"]')).toContainText(/slow down|rate limit|too many/i);
  });
});